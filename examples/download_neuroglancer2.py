# python3 -m examples.download_neuroglancer https://storage.googleapis.com/neuroglancer-fafb-data/fafb_v14/fafb_v14_orig tmp/fafb_v14/color/1
from itertools import product
from functools import partial
from concurrent.futures import ThreadPoolExecutor
import sys
import logging
import json
from PIL import Image
from io import BytesIO
import numpy as np
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import wkw
from wkcuber.utils import ParallelExecutor, time_start, time_stop

CUBE_SIZE = 1024
RETRY_COUNT = 100
RETRY_BACKOFF = 0.3


def download_bucket(prefix, scale_key, offset, bucket_size):
    session = requests.Session()
    retry = Retry(
        total=RETRY_COUNT,
        read=RETRY_COUNT,
        connect=RETRY_COUNT,
        backoff_factor=RETRY_BACKOFF,
        status_forcelist=(500, 502, 504),
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

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
    res = session.get(url)
    if res.status_code != 200:
        logging.info("Empty bucket={}".format(offset))
        return np.zeros(bucket_size, dtype=np.uint8)
    img = Image.open(BytesIO(res.content))
    return np.array(img).reshape(bucket_size).transpose((2, 1, 0))


def download_cube(dataset_path, prefix, scale_key, offset, bucket_size):
    logging.info("Downloading cube={}".format(offset))
    time_start("Done cube={}".format(offset))
    cube_buffer = np.zeros((CUBE_SIZE, CUBE_SIZE, CUBE_SIZE), dtype=np.uint8)
    coords = list(
        product(
            range(CUBE_SIZE // bucket_size[0]),
            range(CUBE_SIZE // bucket_size[1]),
            range(CUBE_SIZE // bucket_size[2]),
        )
    )
    with ThreadPoolExecutor(50) as pool:
        futures = []

        def callback(xyz, buf):
            x, y, z = xyz
            cube_buffer[
                (x * bucket_size[0]) : (x * bucket_size[0] + bucket_size[0]),
                (y * bucket_size[1]) : (y * bucket_size[1] + bucket_size[1]),
                (z * bucket_size[2]) : (z * bucket_size[2] + bucket_size[2]),
            ] = buf.result()
            logging.info("Downloaded bucket={} in cube={}".format((x, y, z), offset))

        for x, y, z in coords:
            bucket_offset = (
                offset[0] + x * bucket_size[0],
                offset[1] + y * bucket_size[1],
                offset[2] + z * bucket_size[2],
            )
            buf_future = pool.submit(
                download_bucket, prefix, scale_key, bucket_offset, bucket_size
            )

            callback_xyz = partial(callback, (x, y, z))
            buf_future.add_done_callback(callback_xyz)
            futures.append(buf_future)
        [f.result() for f in futures]

    if np.all(cube_buffer == 0):
        logging.info("Skipping empty cube={}".format(offset))
        return

    with wkw.Dataset.open(dataset_path) as ds:
        ds.write(offset, cube_buffer)
    time_stop("Done cube={}".format(offset))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    dataset_url = sys.argv[1]
    dataset_info = json.loads(requests.get("{}/info".format(dataset_url)).content)
    dataset_scale_key = dataset_info["scales"][0]["key"]
    dataset_size = dataset_info["scales"][0]["size"]
    dataset_bucket_size = dataset_info["scales"][0]["chunk_sizes"][0]

    target_path = sys.argv[2]
    with open(sys.argv[3], "rt") as cube_file:
        cubes = sorted(json.load(cube_file))

    ds = wkw.Dataset.open(
        target_path,
        wkw.Header(voxel_type=np.uint8, block_type=wkw.Header.BLOCK_TYPE_LZ4),
    )
    ds.close()

    with ParallelExecutor(1) as exec:
        for (cube_x, cube_y, cube_z) in cubes[2000:2001]:
            exec.submit(
                download_cube,
                target_path,
                dataset_url,
                dataset_scale_key,
                (cube_x * CUBE_SIZE, cube_y * CUBE_SIZE, cube_z * CUBE_SIZE),
                dataset_bucket_size,
            )
