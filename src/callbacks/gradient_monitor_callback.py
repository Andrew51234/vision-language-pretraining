from pytorch_lightning.callbacks import Callback
import torch

class GradientMonitorCallback(Callback):
    """
    PyTorch Lightning callback that monitors gradient norms during training.
    
    This callback computes and logs:
    - Overall mean and standard deviation of gradient norms across all parameters
    - Module-specific mean and standard deviation of gradient norms for specified module names
    
    Logged metrics are prefixed with 'gradients/' and include:
    - overall_mean, overall_std: Statistics for all parameters
    - {module_name}_mean, {module_name}_std: Statistics for specific modules
    """

    def __init__(self, names=[]):
        """
        Initialize the gradient monitoring callback.

        Args:
            names (list): List of module name strings to monitor separately.
                         Each parameter containing any of these strings will be grouped
                         and monitored together. Default: []
        """
        super().__init__()
        self.names = names

    def _update_welford(self, existing_stats, new_stats):
        """
        Update mean and variance using Welford's online algorithm.
        
        Args:
            existing_stats: (count, mean, M2) tuple for existing data
            new_stats: (count, mean, variance) tuple for new data
        Returns:
            Updated (count, mean, M2) tuple
        """
        if existing_stats is None:
            n, mean, var = new_stats
            return (n, mean, var * n)
        
        n1, mean1, m2_1 = existing_stats
        n2, mean2, var2 = new_stats
        
        n = n1 + n2
        delta = mean2 - mean1
        mean = mean1 + delta * (n2 / n)
        m2 = m2_1 + n2 * var2 + (delta ** 2) * n1 * n2 / n
        
        return (n, mean, m2)

    def on_after_backward(self, trainer, pl_module):
        """
        Compute and log gradient statistics after backward pass.

        Args:
            trainer: PyTorch Lightning trainer instance
            pl_module: The current PyTorch Lightning module being trained

        The method:
        1. Collects actual gradient values for all parameters
        2. Groups gradients by specified module names
        3. Computes and logs statistics (mean, std) for each group and overall
        """
        module_stats = {name: None for name in self.names}
        all_stats = None
        
        for name, param in pl_module.named_parameters():
            if param.grad is not None:
                # Compute statistics for absolute gradient values
                grad_values = param.grad.abs()  # Take absolute values
                count = grad_values.numel()
                mean = grad_values.mean().item()
                var = grad_values.var().item()
                param_stats = (count, mean, var)
                
                # Update overall statistics
                all_stats = self._update_welford(all_stats, param_stats)
                
                # Update module-specific statistics
                for module_name in self.names:
                    if module_name in name:
                        module_stats[module_name] = self._update_welford(
                            module_stats[module_name], 
                            param_stats
                        )

        # Initialize metrics dictionary
        metric_dict = {}
        
        # Calculate overall statistics
        if all_stats:
            count, mean, m2 = all_stats
            std = (m2 / count) ** 0.5
            metric_dict.update({
                "gradients/overall_mean": mean,
                "gradients/overall_std": std
            })
        
        # Calculate statistics for each module group
        for module_name, stats in module_stats.items():
            if stats:
                count, mean, m2 = stats
                std = (m2 / count) ** 0.5
                metric_dict.update({
                    f"gradients/{module_name}_mean": mean,
                    f"gradients/{module_name}_std": std
                })

        if metric_dict:
            for logger in trainer.loggers:
                logger.log_metrics(
                    metric_dict,
                    step=trainer.fit_loop.epoch_loop.global_step
                )