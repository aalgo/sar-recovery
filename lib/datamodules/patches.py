from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader
from .components.patches import PatchesDataset
from loguru import logger


class PatchesDataModule(LightningDataModule):
    def __init__(
        self,
        input_train_image,
        output_train_image,
        input_test_image, 
        output_test_image,
        input_val_image,
        output_val_image,
        splitmask,
        patch_size,
        shuffle_train = True,
        shuffle_test  = False,
        shuffle_val   = False,
        batch_size: int = 16,
        num_workers: int = 1,
        pin_memory: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.train_dataset = PatchesDataset(
            input_image  = input_train_image,
            output_image = output_train_image,
            splitmask    = splitmask,
            patch_size   = patch_size,
            split        = 'train'
        )

        self.test_dataset = PatchesDataset(
            input_image  = input_test_image,
            output_image = output_test_image,
            splitmask  = splitmask,
            patch_size = patch_size,
            split      = 'test'
        )

        self.val_dataset = PatchesDataset(
            input_image  = input_val_image,
            output_image = output_val_image,
            splitmask  = splitmask,
            patch_size = patch_size,
            split      = 'val'
        )

        logger.info(self.train_dataset)


    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=self.hparams.shuffle_train,
            persistent_workers=True,
            prefetch_factor=8,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=self.hparams.shuffle_val,
            persistent_workers=True,
            prefetch_factor=8,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=self.hparams.shuffle_test,
            persistent_workers=True,
            prefetch_factor=8,
        )