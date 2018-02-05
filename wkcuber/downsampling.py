import time
import logging
import re
import numpy as np
from math import floor
from os import path, listdir
from itertools import product
from scipy.ndimage.interpolation import zoom
from concurrent.futures import ProcessPoolExecutor
from .cube_io import read_cube, write_cube, get_cube_full_path

CUBE_FOLDER_REGEX = re.compile('^[xyz](\d{4})$')


def determine_existing_cube_dims(target_path, mag):

    prefix = path.join(target_path, 'color', str(mag))

    x_dims = [x for x in listdir(prefix) if CUBE_FOLDER_REGEX.match(x)]
    y_dims = [y for y in listdir(path.join(prefix, x_dims[0])) if CUBE_FOLDER_REGEX.match(y)]
    z_dims = [z for z in listdir(path.join(prefix, x_dims[0], y_dims[0])) if CUBE_FOLDER_REGEX.match(z)]

    x_dims.sort()
    y_dims.sort()
    z_dims.sort()

    result = list(product(
        [int(CUBE_FOLDER_REGEX.match(x).group(1)) for x in x_dims],
        [int(CUBE_FOLDER_REGEX.match(x).group(1)) for x in y_dims], 
        [int(CUBE_FOLDER_REGEX.match(x).group(1)) for x in z_dims]))
    result.sort()
    return result


def downsample(config, source_mag, target_mag):

    assert source_mag < target_mag
    logging.info("Downsampling mag {} from mag {}".format(
        target_mag, source_mag))

    factor = int(target_mag / source_mag)
    target_path = config['dataset']['target_path']
    num_downsampling_cores = config['processing']['num_downsampling_cores']

    source_cube_dims = determine_existing_cube_dims(target_path, source_mag)
#    if len(source_cube_dims) <= 1:
#        logging.info("No need to downsample mag {} from mag {}", target_mag, source_mag)
#        return

    cube_coordinates = set(map(lambda xyz: tuple(map(lambda x: floor(x / factor), xyz)), source_cube_dims))

    with ProcessPoolExecutor(num_downsampling_cores) as pool:
        logging.debug("Using up to {} worker processes".format(
            num_downsampling_cores))
        for cube_x, cube_y, cube_z in cube_coordinates:
            pool.submit(downsample_cube_job, config,
                        source_mag, target_mag,
                        cube_x, cube_y, cube_z)


def downsample_cube_job(config, source_mag, target_mag,
                        cube_x, cube_y, cube_z):
    factor = int(target_mag / source_mag)
    dtype = config['dataset']['dtype']
    target_path = config['dataset']['target_path']
    cube_edge_len = config['processing']['cube_edge_len']
    skip_already_downsampled_cubes = config[
        'processing']['skip_already_downsampled_cubes']

    cube_full_path = get_cube_full_path(
        target_path, config['dataset']['name'], target_mag, cube_x, cube_y, cube_z)
    if skip_already_downsampled_cubes and path.exists(cube_full_path):
        logging.debug("Skipping downsampling {},{},{} mag {}".format(
            cube_x, cube_y, cube_z, target_mag))
        return

    logging.debug("Downsampling {},{},{} mag {}".format(
        cube_x, cube_y, cube_z, target_mag))

    ref_time = time.time()
    cube_buffer = np.zeros((cube_edge_len * factor,) * 3, dtype=dtype)
    for local_x in range(factor):
        for local_y in range(factor):
            for local_z in range(factor):
                cube_data = read_cube(
                    target_path, config['dataset']['name'], source_mag, cube_edge_len,
                    cube_x * factor + local_x,
                    cube_y * factor + local_y,
                    cube_z * factor + local_z,
                    dtype)
                cube_buffer[
                    local_x * cube_edge_len:
                    (local_x + 1) * cube_edge_len,
                    local_y * cube_edge_len:
                    (local_y + 1) * cube_edge_len,
                    local_z * cube_edge_len:
                    (local_z + 1) * cube_edge_len
                ] = cube_data

    cube_data = downsample_cube(cube_buffer, factor, dtype)
    write_cube(target_path, config['dataset']['name'], cube_data, target_mag, cube_x, cube_y, cube_z)

    logging.debug("Downsampling took {:.8f}s".format(
        time.time() - ref_time))
    logging.info("Downsampled cube: {},{},{} mag {}".format(
        cube_x, cube_y, cube_z, target_mag))


def downsample_cube(cube_buffer, factor, dtype):
    BILINEAR=1
    BICUBIC=2
    return zoom(
        cube_buffer, 1 / factor, output=dtype,
        # 1: bilinear
        # 2: bicubic
        order=BILINEAR,
        # this does not mean nearest interpolation, it corresponds to how the
        # borders are treated.
        mode='nearest',
        prefilter=True)
