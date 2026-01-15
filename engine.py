import torch
from tqdm.auto import tqdm

def train_step(model, dataloader, loss_fn, optimizer, device):
    """
    Trains a PyTorch model for a single epoch.

    Args:
        model (torch.nn.Module): The PyTorch model to be trained.
        dataloader (DataLoader): A DataLoader instance for the training data.
        loss_fn (torch.nn.Module): The loss function to minimize.
        optimizer (torch.optim.Optimizer): The optimizer to update weights.
        device (torch.device): The device to compute on (e.g., "cuda" or "cpu").

    Returns:
        tuple: (train_loss, train_acc) representing the average metrics for the epoch.
    """
    model.train()
    train_loss, train_acc = 0, 0

    for batch in dataloader:
        # Move data to target device
        X, y = batch[0].to(device), batch[1].to(device)
        
        # Forward pass
        outputs = model(X)
        loss = loss_fn(outputs.float(), y)
        train_loss += loss.item()

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Calculate accuracy
        preds = torch.argmax(outputs, dim=1)
        if y.dtype == torch.float32 or y.dtype == torch.float64:
            y = y.long()
        train_acc += (preds == y).sum().item() / len(y)

    # Average metrics over all batches
    train_loss /= len(dataloader)
    train_acc /= len(dataloader)

    return train_loss, train_acc

def val_step(model, dataloader, loss_fn, device):
    """
    Validates a PyTorch model for a single epoch.

    Args:
        model (torch.nn.Module): The PyTorch model to be validated.
        dataloader (DataLoader): A DataLoader instance for the validation data.
        loss_fn (torch.nn.Module): The loss function to calculate metrics.
        device (torch.device): The device to compute on.

    Returns:
        tuple: (val_loss, val_acc) representing the average metrics for the epoch.
    """
    model.eval() 
    val_loss, val_acc = 0, 0

    with torch.inference_mode():
        for batch in dataloader:
            X, y = batch[0].to(device), batch[1].to(device)
            
            # Forward pass
            outputs = model(X)
            loss = loss_fn(outputs, y)
            val_loss += loss.item()

            # Calculate accuracy
            preds = outputs.argmax(dim=1)
            if y.dtype == torch.float32 or y.dtype == torch.float64:
                y = y.long()
            val_acc += (preds == y).sum().item() / len(y)

    # Average metrics over all batches
    val_loss /= len(dataloader)
    val_acc /= len(dataloader)
    
    return val_loss, val_acc

def train(model, train_dataloader, val_dataloader, optimizer, loss_fn, 
          device, epochs, model_save_path, use_scheduler=False, scheduler=None):
    """
    Trains and validates a model over multiple epochs.

    Saves the model weights that achieve the highest validation accuracy.
    Optionally applies a learning rate scheduler.

    Args:
        model (torch.nn.Module): The model to be trained and validated.
        train_dataloader (DataLoader): DataLoader for the training data.
        val_dataloader (DataLoader): DataLoader for the validation data.
        optimizer (torch.optim.Optimizer): The optimizer for weight updates.
        loss_fn (torch.nn.Module): The loss function to minimize.
        device (torch.device): Target device to run the training on.
        epochs (int): Total number of training epochs.
        model_save_path (str): Path to save the best performing model.
        use_scheduler (bool, optional): Whether to apply a scheduler step. Defaults to False.
        scheduler (optional): A PyTorch learning rate scheduler. Defaults to None.

    Returns:
        tuple: (results_dict, trained_model) where results_dict contains history.
    """
    results = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }
    
    best_val_acc = 0.0
    model.to(device)

    for epoch in tqdm(range(epochs), colour="blue"):
        # Training phase
        train_loss, train_acc = train_step(
            model=model,
            dataloader=train_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device
        )
        
        # Validation phase
        val_loss, val_acc = val_step(
            model=model,
            dataloader=val_dataloader,
            loss_fn=loss_fn,
            device=device
        )
        
        # Log progress
        print(f"Epoch: {epoch+1} | "
              f"train_loss: {train_loss:.4f} | train_acc: {train_acc:.4f} | "
              f"val_loss: {val_loss:.4f} | val_acc: {val_acc:.4f}")
        
        # Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["val_loss"].append(val_loss)
        results["val_acc"].append(val_acc)
        
        # Step the scheduler if applicable
        if use_scheduler and scheduler is not None:
            scheduler.step()
        
        # Checkpointing: Save the best model based on validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), model_save_path)
            print(f" >> Best model saved with accuracy: {best_val_acc:.4f}")

    # Load the best weights back into the model before returning
    print("Loading the best model weights for final output...")
    model.load_state_dict(torch.load(model_save_path, weights_only=True))

    return results, model