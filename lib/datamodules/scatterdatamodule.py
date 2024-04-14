from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader
from .components.scatterdataset import ScatterCoherencePatchesDataset, ScatterPatchesDataset
from loguru import logger


class ScatterCoherencePatchesDataModule(LightningDataModule):
    def __init__(
        self,
        base_path,
        date_train,
        date_test, 
        date_val,
        splitmask_fn_src,
        patch_size,
        avg_window_size,
        scatter_elems=['Svv', 'Svh'],
        coherence_elems=['Shh2'],
        shuffle_train = True,
        shuffle_test  = False,
        shuffle_val   = False,
        batch_size: int = 16,
        num_workers: int = 1,
        pin_memory: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.train_dataset = ScatterCoherencePatchesDataset(
            base_path    = base_path,
            date         = date_train,
            splitmask_fn_src = splitmask_fn_src,
            patch_size   = patch_size,
            scatter_elems = scatter_elems,
            coherence_elems = coherence_elems,
            avg_window_size = avg_window_size,
            split        = 'train'
        )

        self.test_dataset = ScatterCoherencePatchesDataset(
            base_path    = base_path,
            date         = date_test,
            splitmask_fn_src = splitmask_fn_src,
            patch_size   = patch_size,
            scatter_elems = scatter_elems,
            coherence_elems = coherence_elems,
            avg_window_size = avg_window_size,
            split        = 'test'
        )

        self.val_dataset = ScatterCoherencePatchesDataset(
            base_path    = base_path,
            date         = date_val,
            splitmask_fn_src = splitmask_fn_src,
            patch_size   = patch_size,
            scatter_elems = scatter_elems,
            coherence_elems = coherence_elems,
            avg_window_size = avg_window_size,
            split        = 'val'
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

class ScatterPatchesDataModule(LightningDataModule):
    def __init__(
        self,
        base_path,
        date_train,
        date_test, 
        date_val,
        splitmask_fn_src,
        patch_size,
        input_elems=['Svv', 'Svh'],
        output_elems=['Shh'],
        shuffle_train = True,
        shuffle_test  = False,
        shuffle_val   = False,
        batch_size: int = 16,
        num_workers: int = 1,
        pin_memory: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.train_dataset = ScatterPatchesDataset(
            base_path    = base_path,
            date         = date_train,
            splitmask_fn_src = splitmask_fn_src,
            patch_size   = patch_size,
            input_elems  = input_elems,
            output_elems = output_elems,
            split        = 'train'
        )

        self.test_dataset = ScatterPatchesDataset(
            base_path    = base_path,
            date         = date_test,
            splitmask_fn_src = splitmask_fn_src,
            patch_size   = patch_size,
            input_elems  = input_elems,
            output_elems = output_elems,
            split        = 'test'
        )

        self.val_dataset = ScatterPatchesDataset(
            base_path    = base_path,
            date         = date_val,
            splitmask_fn_src = splitmask_fn_src,
            patch_size   = patch_size,
            input_elems  = input_elems,
            output_elems = output_elems,
            split        = 'val'
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