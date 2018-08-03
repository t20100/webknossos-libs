import tarfile
import logging
from argparse import ArgumentParser
from wkcuber.tile_cubing import parse_tile_file_name
from os import path

BUFFER_SIZE = 1024 * 1024
CACHE_DIR = "./tmp"

logging.basicConfig(level=logging.DEBUG)

parser = ArgumentParser()
parser.add_argument("start", type=int)
parser.add_argument("end", type=int)
parser.add_argument("target_path")

args = parser.parse_args()


def detect_coords(z):
    file_name = path.join(CACHE_DIR, "{:04d}.tar".format(z))
    tar = tarfile.open(file_name, bufsize=BUFFER_SIZE)
    for tarinfo in tar:
        if tarinfo.name.startswith("0/") and tarinfo.name.endswith(".jpg"):
            # Decoding image and writing to buffer
            y, x, ext = parse_tile_file_name(tarinfo.name)
            yield (x, y, z, ext, tarinfo.offset_data, tarinfo.size)
            # logging.debug("Found x={} y={} z={}".format(x, y, z))


with open(args.target_path, "w") as f:
    f.write("x,y,z,ext,offset,size\n")
    for z in range(args.start, args.end):
        for tup in sorted(detect_coords(z)):
            f.write("{},{},{},{},{},{}\n".format(*tup))
        f.flush()
