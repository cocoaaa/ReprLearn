import torch
import pytorch_lightning as pl

class HistogramLogger(pl.Callback):
    """
    Distribution of the outputs of the encoder, upon the input of a mini-batch x
    - eg.

    [batch_mu_z, batch_logvar_z] = model.encoder(x)
    Then, plot all elements of batch_mu_z as a historgram. Same for batch_logvar_z

    """

    def __init__(
            self,
            hist_epoch_interval: int = 20,
    ):
        super().__init__()
        self.hist_epoch_interval = hist_epoch_interval

    def on_fit_start(self, *args, **kwargs):
        print(f"{self.__class__.__name__} is called")

    def on_epoch_end(self, trainer, pl_module):
        if (trainer.current_epoch + 1) % self.hist_epoch_interval == 0:
            self.log_hist_of_params_of_q(trainer, pl_module)

    @classmethod
    def log_hist_of_params_of_q(cls, trainer, model, **kwargs):
        is_training = model.training
        model.eval()
        with torch.no_grad():
            for mode in ['train', 'val']:
                dl = getattr(model, f"{mode}_dataloader")()
                x, y = next(iter(dl))
                x = x.to(model.device)

                mu, log_var = model.encode(x)
                var = log_var.exp()
                var_thresh = kwargs.get('var_thresh', None) or 1e-3
                n_tiny_vars = (var < var_thresh).sum()
                p_tiny = n_tiny_vars / log_var.numel()

                # Log the histograms to tensorboard
                trainer.logger.experiment.add_histogram(f"{mode}/centers of q", mu, global_step=model.current_epoch)
                trainer.logger.experiment.add_histogram(f"{mode}/vars of q", var, global_step=model.current_epoch)
                trainer.logger.experiment.add_scalar(f"{mode}/p_tiny_var", p_tiny, global_step=model.current_epoch)
        model.train(is_training)


