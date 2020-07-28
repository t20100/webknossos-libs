import logging
import math

import wkw
import numpy as np
from argparse import ArgumentParser
from scipy.ndimage.interpolation import zoom
from itertools import product
from enum import Enum
from .mag import Mag
from .metadata import refresh_metadata

from .utils import (
    add_verbose_flag,
    open_wkw,
    WkwDatasetInfo,
    ensure_wkw,
    time_start,
    time_stop,
    add_distribution_flags,
    add_interpolation_flag,
    get_executor_for_args,
    wait_and_ensure_success,
    setup_logging,
    cube_addresses,
)

DEFAULT_EDGE_LEN = 256


def determine_buffer_edge_len(dataset):
    return min(DEFAULT_EDGE_LEN, dataset.header.file_len * dataset.header.block_len)


def extend_wkw_dataset_info_header(wkw_info, **kwargs):
    for key, value in kwargs.items():
        setattr(wkw_info.header, key, value)


def calculate_virtual_scale_for_target_mag(target_mag):
    """
    This scale is not the actual scale of the dataset
    The virtual scale is used for downsample_mags_anisotropic.
    """
    max_target_value = max(list(target_mag.to_array()))
    scale_array = max_target_value / np.array(target_mag.to_array())
    return tuple(scale_array)


class InterpolationModes(Enum):
    MEDIAN = 0
    MODE = 1
    NEAREST = 2
    BILINEAR = 3
    BICUBIC = 4
    MAX = 5
    MIN = 6


def create_parser():
    parser = ArgumentParser()

    parser.add_argument("path", help="Directory containing the dataset.")

    parser.add_argument(
        "--layer_name",
        "-l",
        help="Name of the cubed layer (color or segmentation)",
        default="color",
    )

    parser.add_argument(
        "--from_mag",
        "--from",
        "-f",
        help="Resolution to base upsampling on",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--target_mag",
        help="Specify an explicit  target magnification (e.g., --target_mag 16-16-4).",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--buffer_cube_size",
        "-b",
        help="Size of buffered cube to be downsampled (i.e. buffer cube edge length)",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--no_compress",
        help="Don't compress data during downsampling",
        default=False,
        action="store_true",
    )

    add_interpolation_flag(parser)
    add_verbose_flag(parser)
    add_distribution_flags(parser)

    return parser


def upsample(
    source_wkw_info,
    target_wkw_info,
    source_mag: Mag,
    target_mag: Mag,
    interpolation_mode,
    compress,
    buffer_edge_len=None,
    args=None,
):

    assert source_mag > target_mag
    logging.info("Upsampling mag {} from mag {}".format(target_mag, source_mag))

    mag_factors = [
        s // t for (t, s) in zip(target_mag.to_array(), source_mag.to_array())
    ]
    # Detect the cubes that we want to downsample
    source_cube_addresses = cube_addresses(source_wkw_info)

    target_cube_addresses = list(
        set(
            tuple(dim * mag_factor for (dim, mag_factor) in zip(xyz, mag_factors))
            for xyz in source_cube_addresses
        )
    )
    target_cube_addresses.sort()
    with open_wkw(source_wkw_info) as source_wkw:
        if buffer_edge_len is None:
            buffer_edge_len = determine_buffer_edge_len(source_wkw)
        logging.debug(
            "Found source cubes: count={} size={} min={} max={}".format(
                len(source_cube_addresses),
                (buffer_edge_len,) * 3,
                min(source_cube_addresses),
                max(source_cube_addresses),
            )
        )
        logging.debug(
            "Found target cubes: count={} size={} min={} max={}".format(
                len(target_cube_addresses),
                (buffer_edge_len,) * 3,
                min(target_cube_addresses),
                max(target_cube_addresses),
            )
        )

    with open_wkw(source_wkw_info) as source_wkw:
        num_channels = source_wkw.header.num_channels
        header_block_type = (
            wkw.Header.BLOCK_TYPE_LZ4HC if compress else wkw.Header.BLOCK_TYPE_RAW
        )

        extend_wkw_dataset_info_header(
            target_wkw_info,
            num_channels=num_channels,
            file_len=source_wkw.header.file_len,
            block_type=header_block_type,
        )

        ensure_wkw(target_wkw_info)

    with get_executor_for_args(args) as executor:
        job_args = []
        voxel_count_per_cube = (
            source_wkw.header.file_len * source_wkw.header.block_len
        ) ** 3
        job_count_per_log = math.ceil(
            1024 ** 3 / voxel_count_per_cube
        )  # log every gigavoxel of processed data
        for i, target_cube_xyz in enumerate(target_cube_addresses):
            use_logging = i % job_count_per_log == 0

            job_args.append(
                (
                    source_wkw_info,
                    target_wkw_info,
                    mag_factors,
                    interpolation_mode,
                    target_cube_xyz,
                    buffer_edge_len,
                    compress,
                    use_logging,
                )
            )
        wait_and_ensure_success(executor.map_to_futures(upsample_cube_job, job_args))

    logging.info("Mag {0} successfully cubed".format(target_mag))


def upsample_cube_job(args):
    (
        source_wkw_info,
        target_wkw_info,
        mag_factors,
        interpolation_mode,
        target_cube_xyz,
        buffer_edge_len,
        compress,
        use_logging,
    ) = args

    if use_logging:
        logging.info("Upsampling of {}".format(target_cube_xyz))

    try:
        if use_logging:
            time_start("Upsampling of {}".format(target_cube_xyz))
        header_block_type = (
            wkw.Header.BLOCK_TYPE_LZ4HC if compress else wkw.Header.BLOCK_TYPE_RAW
        )

        with open_wkw(source_wkw_info) as source_wkw:
            num_channels = source_wkw.header.num_channels
            source_dtype = source_wkw.header.voxel_type

            extend_wkw_dataset_info_header(
                target_wkw_info,
                voxel_type=source_dtype,
                num_channels=num_channels,
                file_len=source_wkw.header.file_len,
                block_type=header_block_type,
            )

            with open_wkw(target_wkw_info) as target_wkw:
                wkw_cubelength = (
                    source_wkw.header.file_len * source_wkw.header.block_len
                )
                shape = (num_channels,) + (wkw_cubelength,) * 3
                file_buffer = np.zeros(shape, source_dtype)
                tile_length = buffer_edge_len
                tile_count_per_dim = wkw_cubelength // tile_length

                assert (
                    wkw_cubelength % buffer_edge_len == 0
                ), "buffer_cube_size must be a divisor of wkw cube length"

                tile_indices = list(range(0, tile_count_per_dim))
                tiles = product(tile_indices, tile_indices, tile_indices)
                file_offset = wkw_cubelength * np.array(target_cube_xyz)

                for tile in tiles:
                    target_offset = np.array(
                        tile
                    ) * tile_length + wkw_cubelength * np.array(target_cube_xyz)
                    source_offset = target_offset // mag_factors

                    # Read source buffer
                    cube_buffer_channels = source_wkw.read(
                        source_offset,
                        (wkw_cubelength // np.array(mag_factors) // tile_count_per_dim),
                    )

                    for channel_index in range(num_channels):
                        cube_buffer = cube_buffer_channels[channel_index]

                        if not np.all(cube_buffer == 0):
                            # Downsample the buffer

                            data_cube = upsample_cube(
                                cube_buffer, mag_factors, interpolation_mode
                            )

                            buffer_offset = target_offset - file_offset
                            buffer_end = buffer_offset + tile_length

                            file_buffer[
                                channel_index,
                                buffer_offset[0] : buffer_end[0],
                                buffer_offset[1] : buffer_end[1],
                                buffer_offset[2] : buffer_end[2],
                            ] = data_cube

                # Write the downsampled buffer to target
                target_wkw.write(file_offset, file_buffer)
        if use_logging:
            time_stop("Upsampling of {}".format(target_cube_xyz))

    except Exception as exc:
        logging.error("Upsampling of {} failed with {}".format(target_cube_xyz, exc))
        raise exc


def linear_filter_3d(data, factors, order):
    factors = np.array(factors)

    if not np.all(factors == factors[0]):
        logging.debug(
            "the selected filtering strategy does not support anisotropic upsampling. Selecting {} as uniform upsampling factor".format(
                factors[0]
            )
        )
    factor = factors[0]

    ds = data.shape
    assert not any((d % factor > 0 for d in ds))
    return zoom(
        data,
        factor,
        output=data.dtype,
        # 0: nearest
        # 1: bilinear
        # 2: bicubic
        order=order,
        # this does not mean nearest interpolation,
        # it corresponds to how the borders are treated.
        mode="nearest",
        prefilter=True,
    )


def upsample_cube(cube_buffer, factors, interpolation_mode):
    if interpolation_mode == InterpolationModes.NEAREST:
        return linear_filter_3d(cube_buffer, factors, 0)
    elif interpolation_mode == InterpolationModes.BILINEAR:
        return linear_filter_3d(cube_buffer, factors, 1)
    elif interpolation_mode == InterpolationModes.BICUBIC:
        return linear_filter_3d(cube_buffer, factors, 2)
    else:
        raise Exception("Invalid interpolation mode: {}".format(interpolation_mode))


def upsample_mag(
    path,
    layer_name,
    source_mag: Mag,
    target_mag: Mag,
    interpolation_mode="default",
    compress=False,
    buffer_edge_len=None,
    args=None,
):
    interpolation_mode = parse_interpolation_mode(interpolation_mode)

    source_wkw_info = WkwDatasetInfo(path, layer_name, source_mag.to_layer_name(), None)
    with open_wkw(source_wkw_info) as source:
        target_wkw_info = WkwDatasetInfo(
            path,
            layer_name,
            target_mag.to_layer_name(),
            wkw.Header(source.header.voxel_type),
        )

    upsample(
        source_wkw_info,
        target_wkw_info,
        source_mag,
        target_mag,
        interpolation_mode,
        compress,
        buffer_edge_len,
        args,
    )


def parse_interpolation_mode(interpolation_mode):
    if interpolation_mode.upper() == "DEFAULT":
        return InterpolationModes.NEAREST
    else:
        return InterpolationModes[interpolation_mode.upper()]


if __name__ == "__main__":
    args = create_parser().parse_args()
    setup_logging(args)

    source_mag = Mag(args.from_mag)
    target_map = Mag(args.target_mag)

    upsample_mag(
        args.path,
        args.layer_name,
        source_mag,
        target_map,
        args.interpolation_mode,
        not args.no_compress,
        args.buffer_cube_size,
        args,
    )

    refresh_metadata(args.path)
