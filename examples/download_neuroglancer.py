# python3 -m examples.download_neuroglancer https://storage.googleapis.com/neuroglancer-fafb-data/fafb_v14/fafb_v14_orig tmp/fafb_v14/color/1

from PIL import Image
from io import BytesIO
import requests
import wkw
import numpy as np
import json
import sys
from wkcuber.utils import ParallelExecutor
import logging

CUBE_SIZE = 1024


def download_bucket(prefix, scale_key, offset, bucket_size):
    url = "{}/{}/{}-{}_{}-{}_{}-{}".format(
        prefix,
        scale_key,
        offset[0],
        offset[0] + bucket_size[0],
        offset[1],
        offset[1] + bucket_size[1],
        offset[2],
        offset[2] + bucket_size[2],
    )
    res = requests.get(url)
    if res.status_code != 200:
        logging.info("Empty bucket={}".format(offset))
        return np.zeros(bucket_size, dtype=np.uint8)
    img = Image.open(BytesIO(res.content))
    return np.array(img).reshape(bucket_size).transpose((2, 1, 0))


def download_cube(dataset_path, prefix, scale_key, offset, bucket_size):
    logging.info("Downloading cube={}".format(offset))
    cube_buffer = np.zeros((CUBE_SIZE, CUBE_SIZE, CUBE_SIZE), dtype=np.uint8)
    for x in range(CUBE_SIZE // bucket_size[0] // 2):
        for y in range(CUBE_SIZE // bucket_size[1] // 2):
            for z in range(CUBE_SIZE // bucket_size[2] // 2):
                bucket_offset = (
                    offset[0] + x * bucket_size[0],
                    offset[1] + y * bucket_size[1],
                    offset[2] + z * bucket_size[2],
                )
                buf = download_bucket(prefix, scale_key, bucket_offset, bucket_size)
                cube_buffer[
                    (x * bucket_size[0]) : (x * bucket_size[0] + bucket_size[0]),
                    (y * bucket_size[1]) : (y * bucket_size[1] + bucket_size[1]),
                    (z * bucket_size[2]) : (z * bucket_size[2] + bucket_size[2]),
                ] = buf
                logging.info(
                    "Downloaded bucket={} in cube={}".format(bucket_offset, offset)
                )

    if np.all(cube_buffer == 0):
        logging.info("Skipping empty cube={}".format(offset))
        return

    with wkw.Dataset.open(dataset_path) as ds:
        ds.write(offset, cube_buffer)
    logging.info("Written cube={}".format(offset))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    dataset_url = sys.argv[1]
    dataset_info = json.loads(requests.get("{}/info".format(dataset_url)).content)
    dataset_scale_key = dataset_info["scales"][0]["key"]
    dataset_size = dataset_info["scales"][0]["size"]
    dataset_bucket_size = dataset_info["scales"][0]["chunk_sizes"][0]

    target_path = sys.argv[2]

    ds = wkw.Dataset.open(
        target_path,
        wkw.Header(voxel_type=np.uint8, block_type=wkw.Header.BLOCK_TYPE_LZ4),
    )
    ds.close()

    with ParallelExecutor(4) as exec:
        for cube_x in range(121, 122):  # dataset_size[0] // CUBE_SIZE):
            for cube_y in range(65, 66):  # dataset_size[1] // CUBE_SIZE):
                for cube_z in range(3, 4):  # dataset_size[2] // CUBE_SIZE):
                    exec.submit(
                        download_cube,
                        target_path,
                        dataset_url,
                        dataset_scale_key,
                        (cube_x * CUBE_SIZE, cube_y * CUBE_SIZE, cube_z * CUBE_SIZE),
                        dataset_bucket_size,
                    )
