import numpy as np
import wkw
import re
import requests
import tarfile
import logging
import gc
import shutil

from time import time
from PIL import Image
from io import BytesIO, StringIO
from argparse import ArgumentParser
from collections import defaultdict
from os import path, listdir, unlink

from wkcuber.utils import get_regular_chunks, get_chunks
from wkcuber.tile_cubing import parse_tile_file_name

BUFFER_SIZE = 1024 * 1024
BATCH_Z = 1024
CACHE_DIR = "./tmp"

logging.basicConfig(level=logging.DEBUG)

parser = ArgumentParser()
parser.add_argument("start", type=int)
parser.add_argument("end", type=int)
parser.add_argument("target_path")

args = parser.parse_args()


def tar_filename(z):
    return path.join(CACHE_DIR, "{:04d}.tar".format(z))


def read_image(file, dtype=np.uint8):
    this_layer = np.array(Image.open(file), np.dtype(dtype))
    this_layer = this_layer.swapaxes(0, 1)
    return this_layer


def read_image_from_tar(x, y, z):
    file_name = tar_filename(z)
    tar = tarfile.open(file_name, bufsize=BUFFER_SIZE)
    tarinfo = tar.getmember("0/{}/{}/{}.jpg".format(z + 1, y, x))
    with tar.extractfile(tarinfo) as reader:
        return read_image(reader)


def detect_coords(z):
    coords = []
    file_name = path.join(CACHE_DIR, "{:04d}.tar".format(z))
    tar = tarfile.open(file_name, bufsize=BUFFER_SIZE)
    for tarinfo in tar:
        if tarinfo.name.startswith("0/") and tarinfo.name.endswith(".jpg"):
            # Decoding image and writing to buffer
            y, x, ext = parse_tile_file_name(tarinfo.name)
            coords.append((x, y, z))
            # logging.debug("Found x={} y={} z={}".format(x, y, z))
    return coords


for batch in get_regular_chunks(args.start, args.end, BATCH_Z):
    coords = []
    for z in batch:
        coords += detect_coords(z)

    xy_coords = sorted(set([(x, y) for x, y, z in coords]))
    for x, y in xy_coords:
        z_batch = [z for z in batch if (x, y, z) in coords]

        if len(z_batch) == 0:
            continue

        ref_time = time()
        buffer = np.zeros((1024, 1024, BATCH_Z), dtype=np.uint8)
        for z in z_batch:
            buffer[:, :, z - batch[0]] = read_image_from_tar(x, y, z)
            logging.debug("Buffered x={} y={} z={}".format(x, y, z))
        logging.debug(
            "Buffering x={} y={} z={} took {:.8f}s".format(
                x, y, batch[0], time() - ref_time
            )
        )

        with wkw.Dataset.open(args.target_path, wkw.Header(np.uint8)) as ds:
            ref_time = time()
            ds.write((x * 1024, y * 1024, batch[0]), buffer)
            logging.debug(
                "Writing x={} y={} z={} shape={} took {:.8f}s".format(
                    x, y, batch[0], buffer.shape, time() - ref_time
                )
            )
