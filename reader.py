"""Unified reader for GazeHub-style ``.label`` files.

One Dataset serves all four benchmarks. Columns are located by *name* from each
file's header (never hard-coded indices), and the 2D gaze is normalised to
``(yaw, pitch)`` radians:

* MPIIFaceGaze / EYEDIAP / Gaze360 store ``2DGaze`` as ``yaw,pitch``.
* ETH-XGaze stores ``gaze`` as ``pitch,yaw``,
  so it is swapped on read.
"""

from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

DINO_MEAN = (0.485, 0.456, 0.406)
DINO_STD = (0.229, 0.224, 0.225)


def _locate_columns(header):
    """Return (face_index, gaze_index, swap) from a GazeHub label header."""
    columns = [column.lower() for column in header.split()]
    if "face" not in columns:
        raise ValueError(f"No Face column in label header: {header.strip()!r}")
    if "2dgaze" in columns:
        return columns.index("face"), columns.index("2dgaze"), False   # yaw,pitch
    if "gaze" in columns:
        return columns.index("face"), columns.index("gaze"), True       # pitch,yaw
    raise ValueError(f"No 2DGaze/gaze column in label header: {header.strip()!r}")


class GazeDataset(Dataset):
    """Face images with ``(yaw, pitch)`` labels read straight from ``.label`` files.

    ``entries`` is a list of filenames or mappings. A mapping may override
    ``root``, ``image_dir``, and ``prefix`` per label file, which is all that a
    cross-dataset experiment needs -- no dataset-specific reader.
    """

    def __init__(self, data_root, image_dir, entries, image_size=None, resize=False):
        if image_size is not None and (int(image_size) != image_size or image_size < 1):
            raise ValueError('image_size must be a positive integer.')
        if resize and image_size is None:
            raise ValueError('Resizing requires image_size.')
        self.image_size, self.resize = image_size, resize
        self.records = []                       # (image_path: str, yaw, pitch)
        for entry in entries:
            entry = entry if isinstance(entry, dict) else {"label": entry}
            root = Path(entry.get("root", data_root)).expanduser()
            base = root / entry.get("image_dir", image_dir) / entry.get("prefix", "")
            with (root / "Label" / entry["label"]).open(encoding="utf-8") as file:
                face_index, gaze_index, swap = _locate_columns(file.readline())
                for line in file:
                    fields = line.split()
                    if not fields:
                        continue
                    first, second = (float(v) for v in fields[gaze_index].split(","))
                    yaw, pitch = (second, first) if swap else (first, second)
                    self.records.append((str(base / fields[face_index]), yaw, pitch))

        if not self.records:
            raise ValueError("No samples were read from the supplied label entries.")
        steps = []
        if resize:
            steps.append(transforms.Resize((image_size, image_size),
                         interpolation=InterpolationMode.BILINEAR, antialias=True))
        self.transform = transforms.Compose(steps + [
            transforms.ToTensor(),
            transforms.Normalize(DINO_MEAN, DINO_STD),
        ])

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        path, yaw, pitch = self.records[index]
        with Image.open(path) as source:
            if (self.image_size is not None and not self.resize
                    and source.size != (self.image_size, self.image_size)):
                raise ValueError(f'Unexpected image size {source.size}: {path}')
            image = self.transform(source.convert("RGB"))
        gaze = torch.tensor((yaw, pitch), dtype=torch.float32)
        return image, gaze, path


def build_loader(data_root, image_dir, entries, batch_size, *, train=False,
                 num_workers=6, seed=None, limit=0, image_size=None, resize=False):
    dataset = GazeDataset(data_root, image_dir, entries, image_size=image_size, resize=resize)
    if limit:
        dataset = Subset(dataset, range(min(limit, len(dataset))))
    generator = torch.Generator().manual_seed(seed) if seed is not None else None
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        generator=generator,
    )
