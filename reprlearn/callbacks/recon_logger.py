import torch
import torchvision
import pytorch_lightning as pl
from reprlearn.data.transforms.functional import unnormalize


class ReconLogger(pl.Callback):

    def __init__(
            self,
            recon_epoch_interval: int = 20,
            normalize: bool = False
    ):
        super().__init__()
        self.recon_epoch_interval = recon_epoch_interval
        self.normalize = normalize

    def on_fit_start(self, *args, **kwargs):
        print(f"{self.__class__.__name__} is called")

    def on_epoch_end(self, trainer, pl_module):
        if (trainer.current_epoch + 1) % self.recon_epoch_interval == 0:
            self.log_recon(trainer, pl_module)

    @classmethod
    def log_recon(cls, trainer, model, **kwargs):
        """
        Run it after train/val epoch
        kwargs:
        - dim (int):
            - Use -1 for side-by-side stacking of x and x_recon.
            - Use -2 for up-down stacking of x and x_recon
        """
        is_training = model.training
        model.eval()

        train_mean, train_std = trainer.datamodule.train_mean, trainer.datamodule.train_std
        with torch.no_grad():
            for mode in ['train', 'val']:
                dl = getattr(model, f"{mode}_dataloader")()
                x, y = next(iter(dl))
                x = x.to(model.device)
                x_recon = model.generate(x)

                # unnormalize for visualization
                x = x.cpu()
                x_recon = x_recon.cpu()
                x_unnormed = unnormalize(x, train_mean, train_std)
                x_recon_unnormed = unnormalize(x_recon, train_mean, train_std)

                # Log input-recon grid to TB
                input_grid = torchvision.utils.make_grid(x_unnormed)  # (C, gridh, gridw)
                recon_grid = torchvision.utils.make_grid(x_recon_unnormed)  # (C, gridh, gridw)
                grid = torch.cat([input_grid, recon_grid], dim=-1)  # inputs | recons
                trainer.logger.experiment.add_image(f"{mode}/recons", grid, global_step=model.current_epoch)
        model.train(is_training)

