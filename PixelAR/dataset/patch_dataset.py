import logging
from typing import Sequence

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from PixelAR.dataset.imagenet import CustomDataset
from PixelAR.models.patcher import Patcher


class PatchDataset(Dataset):
    """
    Iterates over the underlying dataset, computing patch lengths with `patcher`
    and yielding pre-batched triples: (x_batch, y_batch, patch_len_batch).

    If `shuffle=True`, indices are randomly permuted before sharding.
    """

    def __init__(
        self,
        dataset: CustomDataset,
        predict_eoi_token: bool = False,
        patcher: Patcher | None = None,
        attn_impl: str = "xformers",
    ):
        super().__init__()
        self.dataset = dataset
        self.predict_eoi_token = predict_eoi_token
        self.patcher = patcher
        self.attn_impl = attn_impl       

        # logger
        self.logger = logging.Logger(self.__class__.__name__)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # fetch a datum
        datum = self.dataset[idx]  # type: ignore

        x, y = datum[0], datum[1]
        if len(datum) == 3 and self.patcher is not None:
            entropies_i = datum[2].clone()  # entropies
            if self.predict_eoi_token:
                patch_lens: torch.Tensor = self.patcher(
                    tokens=x.reshape(1, -1)[:, :-1],
                    include_next_token=True,
                    entropies=entropies_i.reshape(1, -1),
                )[0][0].clone()  # patch lengths
                patch_lens_w_next: torch.Tensor = self.patcher(
                    tokens=x.reshape(1, -1)[:, :-1],
                    include_next_token=True,
                    entropies=entropies_i.reshape(1, -1),
                    include_eoi_token=self.predict_eoi_token,
                )[0][0].clone()  # patch lengths
            else:
                patch_lens: torch.Tensor = self.patcher(
                    tokens=x.reshape(1, -1)[:, :-1],
                    include_next_token=False,
                    entropies=entropies_i.reshape(1, -1),
                )[0][0].clone()  # patch lengths
                patch_lens_w_next: torch.Tensor = self.patcher(
                    tokens=x.reshape(1, -1)[:, :-1],
                    include_next_token=True,
                    entropies=entropies_i.reshape(1, -1),
                )[0][0].clone()  # patch lengths
            return x, y, patch_lens, patch_lens.numel(), patch_lens_w_next, patch_lens_w_next.numel()
        else:
            return x, y


def _to_scalar(value: torch.Tensor | int) -> int:
        if isinstance(value, torch.Tensor):
            return int(value.item())
        return int(value)

def collate_fn(
    batch: Sequence[tuple[torch.Tensor, torch.Tensor | int, torch.Tensor, int, torch.Tensor, int]], packed_input: bool = False,
):
    """Pack per-example tensors into batch tensors.

    Assumes each item provides token ids, label, patch lengths and the
    corresponding sequence length. Patch lengths and patch sequence lengths are
    concatenated into flat 1D tensors to support packed input processing.
    """
    tokens = torch.stack([example[0] for example in batch], dim=0)
    labels = torch.tensor([_to_scalar(example[1]) for example in batch], dtype=torch.long)
    if packed_input:
        patch_lengths = torch.cat([example[2].reshape(-1) for example in batch], dim=0)
        patch_lengths_w_next = torch.cat([example[4].reshape(-1) for example in batch], dim=0)
        patch_seqlens = torch.tensor([int(example[3]) for example in batch], dtype=torch.long)
        patch_seqlens_w_next = torch.tensor([example[5] for example in batch], dtype=torch.long)
        return tokens, labels, patch_lengths, patch_lengths_w_next, patch_seqlens, patch_seqlens_w_next
    else:
        patch_lens_w_next = pad_sequence([example[4].reshape(-1) for example in batch], batch_first=True, padding_value=0)
        return tokens, labels, patch_lens_w_next
