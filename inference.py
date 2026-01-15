import torch
import torch.nn.functional as F
import rasterio
import numpy as np
from rasterio.windows import Window
from tqdm import tqdm
from PIL import Image

class GeoInference:
    """
    Handles sliding-window inference on large-scale geospatial GeoTIFF images.

    This class breaks down large satellite images into manageable tiles, 
    performs model inference, and reconstructs the results into a georeferenced 
    classification map with overlapping probability averaging.

    Attributes:
        model (torch.nn.Module): The trained PyTorch model.
        transform (callable): Image transformations to apply to tiles.
        device (torch.device): Device for computation (CPU/GPU).
        tile_size (int): The pixel size of each square tile.
        stride (int): The step size between tiles (enabling overlap).
        batch_size (int): Number of tiles to process simultaneously.
    """

    def __init__(self, model, transform, device, tile_size=64, stride=32, batch_size=16):
        """Initializes the GeoInference engine with model and window parameters."""
        self.model = model
        self.transform = transform
        self.device = device
        self.tile_size = tile_size
        self.stride = stride
        self.batch_size = batch_size

    def predict_sliding_window(self, tif_path):
        """
        Runs sliding window inference and returns prediction, confidence scores, and profile.

        Args:
            tif_path (str): Path to the input GeoTIFF file.

        Returns:
            tuple: (prediction_map, final_score_map, profile)
                - prediction_map: 2D array of class indices (0-9, 255 for NoData).
                - final_score_map: 3D array of average probabilities per class.
                - profile: The rasterio profile for saving the output map.
        """
        with rasterio.open(tif_path) as src:
            height, width = src.height, src.width
            profile = src.profile.copy()
            
            # 10 classes for EuroSAT. 
            # score_map accumulates probabilities, count_map tracks overlaps for averaging.
            score_map = np.zeros((10, height, width), dtype=np.float32)
            count_map = np.zeros((height, width), dtype=np.float32)

            # Generate windows for the sliding window grid
            windows = []
            for col_off in range(0, width, self.stride):
                for row_off in range(0, height, self.stride):
                    window = Window(col_off, row_off, self.tile_size, self.tile_size)
                    windows.append(window)

            for i in tqdm(range(0, len(windows), self.batch_size), desc="Inferencing"):
                batch_windows = windows[i : i + self.batch_size]
                batch_imgs = []
                valid_windows = []

                for win in batch_windows:
                    # Read RGB bands (1, 2, 3)
                    data = src.read([1, 2, 3], window=win, boundless=True, fill_value=0)
                    
                    # Skip empty tiles (black background) to save compute
                    if np.all(data == 0): 
                        continue

                    # Prep image for model (CHW -> HWC for PIL)
                    img = np.moveaxis(data, 0, -1).astype(np.uint8)
                    img = Image.fromarray(img)
                    batch_imgs.append(self.transform(img))
                    valid_windows.append(win)

                if not batch_imgs: 
                    continue

                # Run model inference on the batch
                input_tensor = torch.stack(batch_imgs).to(self.device)
                with torch.no_grad():
                    outputs = self.model(input_tensor)
                    probs = F.softmax(outputs, dim=1).cpu().numpy()

                # Map patch probabilities back to the global coordinate system
                for win, prob in zip(valid_windows, probs):
                    row_start, col_start = int(win.row_off), int(win.col_off)
                    h_end = min(row_start + self.tile_size, height)
                    w_end = min(col_start + self.tile_size, width)

                    for c in range(10):
                        score_map[c, row_start:h_end, col_start:w_end] += prob[c]
                    count_map[row_start:h_end, col_start:w_end] += 1

            # --- FINALIZATION ---
            
            # Calculate Average Probability (resolves overlaps)
            final_score_map = score_map / np.maximum(count_map, 1)
            
            # Get the Class with Highest Average Probability per pixel
            prediction_map = np.argmax(final_score_map, axis=0).astype(np.uint8)
            
            # Clean up boundaries: Apply mask based on original RGB data
            original_data_mask = src.read(1) == 0 
            prediction_map[original_data_mask] = 255
            
            return prediction_map, final_score_map, profile