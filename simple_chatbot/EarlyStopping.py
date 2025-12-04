import numpy as np
from typing import Tuple, Dict, Optional

class EarlyStopping:
    """
    Early stopping handler to stop training when a monitored metric has stopped improving.
    
    This implementation provides a clean interface by returning decision flags
    instead of requiring direct attribute access.
    
    Args:
        monitor (str): Metric name to monitor (e.g., 'val_loss', 'val_accuracy')
        mode (str): One of 'min' or 'max'. In 'min' mode, training will stop when the quantity 
                   monitored has stopped decreasing; in 'max' mode it will stop when the quantity 
                   monitored has stopped increasing.
        patience (int): Number of epochs with no improvement after which training will be stopped.
        min_delta (float): Minimum change in the monitored quantity to qualify as an improvement.
        verbose (bool): If True, prints progress messages.
    
    Attributes:
        best_score (float): Best score achieved so far
        best_epoch (int): Epoch number where best score was achieved
        epochs_no_improve (int): Counter for epochs without improvement
    
    Example:
        >>> early_stopping = EarlyStopping(monitor='val_loss', patience=10)
        >>> for epoch in range(epochs):
        >>>     # Training phase
        >>>     train_loss = train_one_epoch(model, train_loader)
        >>>     
        >>>     # Validation phase
        >>>     val_loss = validate(model, val_loader)
        >>>     
        >>>     # Update early stopping - returns two flags
        >>>     should_stop, model_improved = early_stopping(val_loss, epoch=epoch)
        >>>     
        >>>     # Save best model if current epoch shows improvement
        >>>     if model_improved:
        >>>         torch.save(model.state_dict(), "best_model.pth")
        >>>     
        >>>     # Stop training if early stopping condition is met
        >>>     if should_stop:
        >>>         break
        >>> 
        >>> # Get final results
        >>> best_info = early_stopping.get_best_info()
        >>> print(f"Best {best_info['monitor']}: {best_info['best_score']:.4f} at epoch {best_info['best_epoch']}")
    """
    
    def __init__(self, monitor: str = 'val_loss', mode: str = 'min', 
                 patience: int = 10, min_delta: float = 0.001, verbose: bool = True):
        # Store configuration parameters
        self.monitor = monitor
        self.mode = mode
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose

        # Initialize internal state
        self.best_score = None      # Best metric value achieved so far
        self.epochs_no_improve = 0  # Counter for epochs without improvement
        self.best_epoch = None      # Epoch number where best score was achieved

        # Set comparison operation based on mode
        # For 'min' mode: improvement when current < best - min_delta
        # For 'max' mode: improvement when current > best + min_delta
        if self.mode == 'min':
            self._is_improvement = lambda current, best: current < best - self.min_delta
            self.best_score = np.Inf  # Initialize with infinity for minimization
        elif self.mode == 'max':
            self._is_improvement = lambda current, best: current > best + self.min_delta
            self.best_score = -np.Inf  # Initialize with -infinity for maximization
        else:
            raise ValueError(f"Mode '{mode}' is unknown. Use 'min' or 'max'.")

        if self.verbose:
            print(f"EarlyStopping: Monitoring '{monitor}' in '{mode}' mode with patience={patience}")

    def __call__(self, current_score: float, epoch: Optional[int] = None) -> Tuple[bool, bool]:
        """
        Update early stopping state with current metric score and return decision flags.
        
        This method should be called at the end of each epoch with the current value
        of the monitored metric.
        
        Args:
            current_score (float): Current value of the monitored metric
            epoch (int, optional): Current epoch number for logging purposes
            
        Returns:
            Tuple[bool, bool]: 
                - should_stop (bool): True if training should stop due to early stopping
                - model_improved (bool): True if current epoch achieved the best score so far
        """
        model_improved = False  # Flag indicating if current epoch has best score
        should_stop = False     # Flag indicating if training should stop

        # First epoch initialization
        if self.best_score is None:
            self.best_score = current_score
            self.best_epoch = epoch
            self.epochs_no_improve = 0
            model_improved = True
            
            if self.verbose:
                print(f"[Epoch {epoch}] Initial best {self.monitor}: {self.best_score:.6f}")
            return should_stop, model_improved

        # Check if current score shows improvement over best score
        if self._is_improvement(current_score, self.best_score):
            # Improvement detected - update best score and reset counter
            self.best_score = current_score
            self.best_epoch = epoch
            self.epochs_no_improve = 0
            model_improved = True
            
            if self.verbose:
                # Calculate improvement amount for logging
                if self.mode == 'min':
                    improvement = self.best_score - current_score
                else:
                    improvement = current_score - self.best_score
                print(f"[Epoch {epoch}] Improved {self.monitor}: {self.best_score:.6f} (improvement: {improvement:+.6f})")
        else:
            # No improvement detected - increment counter
            self.epochs_no_improve += 1
            
            if self.verbose:
                print(f"[Epoch {epoch}] No improvement in {self.monitor} for {self.epochs_no_improve}/{self.patience} epochs. "
                      f"Current: {current_score:.6f}, Best: {self.best_score:.6f}")
            
            # Check if early stopping condition is met
            if self.epochs_no_improve >= self.patience:
                should_stop = True
                if self.verbose:
                    print(f"[Epoch {epoch}] Early stopping triggered. "
                          f"Best {self.monitor}: {self.best_score:.6f} at epoch {self.best_epoch}")

        return should_stop, model_improved

    def get_best_info(self) -> Dict[str, Optional[float]]:
        """
        Get information about the best performance achieved during training.
        
        Returns:
            Dict[str, Optional[float]]: Dictionary containing:
                - 'best_score': Best metric value achieved
                - 'best_epoch': Epoch number where best score was achieved  
                - 'monitor': Name of the monitored metric
        """
        return {
            'best_score': self.best_score,
            'best_epoch': self.best_epoch,
            'monitor': self.monitor
        }

    def reset(self) -> None:
        """
        Reset the early stopping state to initial conditions.
        
        This is useful when you want to reuse the same EarlyStopping instance
        for multiple training sessions.
        """
        self.best_score = None
        self.epochs_no_improve = 0
        self.best_epoch = None
        
        if self.verbose:
            print("EarlyStopping state has been reset.")