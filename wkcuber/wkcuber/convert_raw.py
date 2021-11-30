from argparse import ArgumentParser, Namespace
import logging
import os.path
from pathlib import Path
import re
import time
from typing import Optional, Tuple
import numpy as np

from webknossos.dataset.defaults import DEFAULT_WKW_FILE_LEN
from wkcuber.api.dataset import Dataset
from wkcuber.utils import (
    add_scale_flag,
    add_verbose_flag,
    parse_shape,
    setup_logging,
)


logger = logging.getLogger(__name__)


def create_parser() -> ArgumentParser:
    parser = ArgumentParser()

    parser.add_argument(
        "source_path",
        help="Path to raw file to convert",
        type=Path,
    )

    parser.add_argument(
        "target_path", help="Output directory for the generated WKW dataset.", type=Path
    )

    parser.add_argument(
        "--layer_name",
        "-l",
        help="Name of the cubed layer (color or segmentation).",
        default="color",
    )

    parser.add_argument(
        "--dtype",
        "-d",
        help="Target datatype (e.g. uint8, uint16, float32)."
        "If not provided, a guess is made from file size and data shape",
        default=None,
    )

    parser.add_argument(
        "--shape",
        help="Shape of the dataset (depth, height, width)",
        type=parse_shape,
    )

    add_scale_flag(parser, required=False)
    add_verbose_flag(parser)

    return parser


def get_shape_from_vol_info_file(filepath: Path) -> Optional[Tuple[int, int, int]]:
    """Check for a <filename>.info file and try to parse it."""
    info_filepath = filepath.with_suffix(filepath.suffix + ".info")
    if not info_filepath.is_file():
        logger.debug(f"No {info_filepath} file found")
        return None

    dims = {}
    regexp = re.compile(r"\s*(?P<key>NUM_[XYZ])\s*=\s*(?P<value>\d+).*")
    try:
        f = info_filepath.open("r")
    except OSError:
        logger.warning(f"Cannot open {info_filepath} file")
        return None

    for line in f:
        match = regexp.match(line)
        if match is not None:
            dims[match.group('key')] = int(match.group('value'))
    f.close()

    missings = set(("NUM_X", "NUM_Y", "NUM_Z")) - set(dims.keys())
    if missings:
        logger.warning(f"Missing {missings} keys in {info_filepath}")
        return None

    return dims["NUM_Z"], dims["NUM_Y"], dims["NUM_Z"]


def get_dtype_from_file_size(filepath: Path, shape: Tuple[int, ...]) -> Optional[str]:
    """Return a data type from given shape and file size.

    Returns None if there is no matching dtype.
    """
    file_size = os.path.getsize(filepath)
    if file_size % np.prod(shape) != 0:
        return None

    itemsize = file_size // np.prod(shape)
    return {1: "uint8", 2: "uint16", 4: "float32", 8: "float64"}.get(itemsize, None)


def convert_raw(
    source_raw_path: Path,
    target_path: Path,
    layer_name: str,
    dtype: Optional[str] = None,
    shape: Optional[Tuple[int, int, int]] = None,
    scale: Optional[Tuple[float, float, float]] = (1.0, 1.0, 1.0),
    file_len: int = DEFAULT_WKW_FILE_LEN,
) -> None:
    ref_time = time.time()

    if shape is None:
        shape = get_shape_from_vol_info_file(source_raw_path)
        if shape is None:
            logger.error("No shape provided and cannot guess it")
            return
        logger.info(f"Using data shape: {shape}")

    if dtype is None:
        dtype = get_dtype_from_file_size(source_raw_path, shape)
        if dtype is None:
            logger.error("No dtype provided and cannot guess it")
            return
        logger.info(f"Using dtype: {dtype}")

    cube_data = np.memmap(source_raw_path, dtype=dtype, mode="r", shape=shape)

    if scale is None:
        scale = 1.0, 1.0, 1.0
    wk_ds = Dataset.get_or_create(target_path, scale=scale)
    wk_layer = wk_ds.get_or_add_layer(
        layer_name,
        "color",
        dtype_per_layer=np.dtype(dtype),
        num_channels=1,
    )
    wk_mag = wk_layer.get_or_add_mag("1", file_len=file_len)
    wk_mag.write(cube_data)

    logger.debug(
        "Converting of {} took {:.8f}s".format(source_raw_path, time.time() - ref_time)
    )


def main(args: Namespace) -> None:
    source_path = args.source_path

    if source_path.is_dir():
        logger.error("source_path is not a file")
        return

    convert_raw(
        source_path,
        args.target_path,
        args.layer_name,
        args.dtype,
        args.shape,
        args.scale,
    )


if __name__ == "__main__":
    args = create_parser().parse_args()
    setup_logging(args)

    main(args)
