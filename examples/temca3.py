import numpy as np
import wkw
import re
import requests
import tarfile
import logging
import gc
import io
import shutil

from time import time
from PIL import Image
from io import BytesIO, StringIO
from argparse import ArgumentParser
from collections import defaultdict, namedtuple
from os import path, listdir, unlink

from wkcuber.utils import get_regular_chunks, get_chunks
from wkcuber.tile_cubing import parse_tile_file_name

BUFFER_SIZE = 1024 * 1024
BATCH_Z = 1024
CACHE_DIR = "./tmp"

CoordInfo = namedtuple("CoordInfo", ["x", "y", "z", "ext", "offset", "size"])

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


def read_image_from_tar(x, y, z, offset, size):
    file_name = tar_filename(z)
    with open(file_name, "rb") as f:
        f.seek(offset)
        b = f.read(size)
        return read_image(io.BytesIO(b))


def detect_coords(z):
    file_name = path.join(CACHE_DIR, "{:04d}.tar".format(z))
    tar = tarfile.open(file_name, bufsize=BUFFER_SIZE)
    for tarinfo in tar:
        if tarinfo.name.startswith("0/") and tarinfo.name.endswith(".jpg"):
            # Decoding image and writing to buffer
            y, x, ext = parse_tile_file_name(tarinfo.name)
            yield CoordInfo(x, y, z, ext, tarinfo.offset_data, tarinfo.size)
            # logging.debug("Found x={} y={} z={}".format(x, y, z))


with wkw.Dataset.open(args.target_path, wkw.Header(np.uint8)) as ds:
    for batch in get_regular_chunks(args.start, args.end, BATCH_Z):
        coords = []
        for z in batch:
            coords += sorted(detect_coords(z))

        xy_coords = sorted(set([(tup.x, tup.y) for tup in coords]))
        coord_map = {(tup.x, tup.y, tup.z): tup for tup in coords}

        for i, (x, y) in enumerate(xy_coords):
            ref_time = time()
            z_batch = [z for z in batch if (x, y, z) in coord_map]

            if len(z_batch) == 0:
                continue

            logging.debug(
                "Stuff took {:.8f}s".format(
                    time() - ref_time
                )
            )

            ref_time = time()
            buffer = np.zeros((1024, 1024, BATCH_Z), dtype=np.uint8)
            for z in z_batch:
                tup = coord_map[(x, y, z)]
                buffer[:, :, z - batch[0]] = read_image_from_tar(
                    x, y, z, tup.offset, tup.size
                )
                logging.debug("Buffered x={} y={} z={}".format(x, y, z))
            logging.debug(
                "Buffering x={} y={} z={} {}/{} took {:.8f}s".format(
                    x, y, batch[0], i, len(xy_coords), time() - ref_time
                )
            )

            ref_time = time()
            ds.write((x * 1024, y * 1024, batch[0]), buffer)
            logging.debug(
                "Writing x={} y={} z={} shape={} {}/{} took {:.8f}s".format(
                    x, y, batch[0], buffer.shape, i, len(xy_coords), time() - ref_time
                )
            )
