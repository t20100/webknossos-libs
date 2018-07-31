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
CACHE_DIR = "./tmp"

logging.basicConfig(level=logging.DEBUG)

parser = ArgumentParser()
parser.add_argument(
    "--skip_download", help="Skip downloading", default=False, action="store_true"
)
parser.add_argument(
    "--clear_files", help="Clears tar files on-disk", default=False, action="store_true"
)
parser.add_argument("--batch_xy", default=1024, type=int)
parser.add_argument("--batch_z", default=32, type=int)
parser.add_argument("start", type=int)
parser.add_argument("end", type=int)
parser.add_argument("target_path")

args = parser.parse_args()


def truncate_folder(folder_path):
    for file_object in listdir(folder_path):
        file_object_path = path.join(folder_path, file_object)
        if path.isfile(file_object_path):
            unlink(file_object_path)
        else:
            shutil.rmtree(file_object_path)


def download(url, file_name):
    r = requests.get(url, stream=True)
    with open(file_name, "wb") as f:
        for chunk in r.iter_content(chunk_size=BUFFER_SIZE):
            if chunk:
                f.write(chunk)


def read_image(file, dtype=np.uint8):
    this_layer = np.array(Image.open(file), np.dtype(dtype))
    this_layer = this_layer.swapaxes(0, 1)
    return this_layer


def detect_coords(z):
    coords = []
    file_name = path.join(CACHE_DIR, "{:04d}.tar".format(z))
    tar = tarfile.open(file_name, bufsize=BUFFER_SIZE)
    for tarinfo in tar:
        if tarinfo.name.startswith("0/") and tarinfo.name.endswith(".jpg"):
            # Decoding image and writing to buffer
            y, x, ext = parse_tile_file_name(tarinfo.name)
            coords.append((x, y, z))
            logging.debug("Found x={} y={} z={}".format(x, y, z))
    return coords


def download_files(batch):
    ref_time = time()
    coords = []

    for z in batch:
        # Download file
        url = "http://temca2data.janelia.org/v14/temca2.14.0.{}.tar".format(z + 1)
        file_name = path.join(CACHE_DIR, "{:04d}.tar".format(z))
        download(url, file_name)
        logging.debug("Loaded z={}".format(z))

        tar = tarfile.open(file_name, bufsize=BUFFER_SIZE)
        for tarinfo in tar:
            if tarinfo.name.startswith("0/") and tarinfo.name.endswith(".jpg"):
                # Decoding image and writing to buffer
                y, x, ext = parse_tile_file_name(tarinfo.name)
                coords.append((x, y, z))
                logging.debug("Found x={} y={} z={}".format(x, y, z))
    logging.info(
        "Downloading z={}-{} took {:.8f}s".format(
            batch[0], batch[-1], time() - ref_time
        )
    )
    return coords


def buffer_tiles(batch, xy_batch):
    ref_time = time()
    buffer = defaultdict(lambda: np.zeros((1024, 1024, args.batch_z), dtype=np.uint8))
    for z in batch:
        file_name = path.join(CACHE_DIR, "{:04d}.tar".format(z))
        tar = tarfile.open(file_name, bufsize=BUFFER_SIZE)
        for tarinfo in tar:
            file_parts = parse_tile_file_name(tarinfo.name)
            if file_parts is not None:
                y, x, ext = file_parts
                if (x, y) in xy_batch:
                    reader = tar.extractfile(tarinfo)
                    img_buf = buffer[(x, y, batch[0])]
                    img = read_image(reader)
                    img_buf[:, :, z - batch[0]] = img
                    logging.debug(
                        "Buffered x={} y={} z={} in={}".format(x, y, z, z - batch[0])
                    )
    logging.info(
        "Buffering z={}-{} batch={} took {:.8f}s".format(
            batch[0], batch[-1], i, time() - ref_time
        )
    )
    return buffer


def write_buffers(buffer):
    ref_time = time()
    # Write buffer to WKW file
    with wkw.Dataset.open(args.target_path, wkw.Header(np.uint8)) as ds:
        for (x, y, z), buf in buffer.items():
            # if x > 110 and y > 110 and x < 113 and y < 113:
            logging.debug(
                "Write buffer x={} y={} z={} shape={}".format(x, y, z, buf.shape)
            )
            ds.write((x * 1024, y * 1024, z), buf)

    logging.info(
        "Writing z={}-{} batch={} took {:.8f}s".format(
            batch[0], batch[-1], i, time() - ref_time
        )
    )


for batch in get_regular_chunks(args.start, args.end, args.batch_z):
    coords = []
    if args.skip_download:
        for z in batch:
            coords += detect_coords(z)
    else:
        coords = download_files(batch)

    xy_coords = sorted(set([(x, y) for x, y, z in coords]))
    for i, xy_batch in enumerate(get_chunks(xy_coords, args.batch_xy)):
        buffer = buffer_tiles(batch, xy_batch)
        write_buffers(buffer)

    if args.clear_files:
        truncate_folder(CACHE_DIR)
