from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from lib.utils.dist import is_main_process
from . import imagenet_loader

_DATA_LOADERS = {
    "imagenet": imagenet_loader.create_dataloader,
}


_PREFETCHER = {
    "imagenet": imagenet_loader.DataPrefetcher,
}


def create_dataloader(name, verbose=True, **kwargs):
    label_only = kwargs.get('label_only', False)
    dataset, _data_loader = _DATA_LOADERS[name](**kwargs)

    def inf_data_loader():
        epoch = 0
        while True:
            if (hasattr(_data_loader, 'sampler') and
                isinstance(_data_loader.sampler, DistributedSampler)
            ):
                _data_loader.sampler.set_epoch(epoch)
            for data in tqdm(_data_loader, disable=not is_main_process() or not verbose, desc="epoch"):
                yield data

            epoch += 1

    if label_only:
        return dataset, inf_data_loader()

    prefetcher = _PREFETCHER[name](
        inf_data_loader(),
        dataset.mean,
        dataset.std
    )

    return dataset, prefetcher
