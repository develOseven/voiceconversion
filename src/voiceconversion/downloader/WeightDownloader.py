import asyncio
import os

from voiceconversion.downloader.Downloader import download
import logging
from voiceconversion.Exceptions import PretrainDownloadException


CREPE_ONNX_FULL: str = "crepe_onnx_full.onnx"
CREPE_ONNX_TINY: str = "crepe_onnx_tiny.onnx"
CREPE_FULL: str = "crepe_full.pth"
CREPE_TINY: str = "crepe_tiny.pth"
CONTENT_VEC_500_ONNX: str = "content_vec_500.onnx"
RMVPE: str = "rmvpe.pt"
RMVPE_ONNX: str = "rmvpe.onnx"
FCPE: str = "fcpe.pt"
FCPE_ONNX: str = "fcpe.onnx"

logger = logging.getLogger(__name__)


async def downloadWeight(
    pretrained_weights_dir: str, hash_cache_file: str | None = None
):
    logger.info("Loading weights.")
    file_params = [
        {
            "url": "https://huggingface.co/wok000/weights/resolve/7a376af24f4f21f9d9e24160ed9d858e8a33bf93/crepe/onnx/full.onnx",
            "saveTo": os.path.join(pretrained_weights_dir, CREPE_ONNX_FULL),
            "hash": "e9bb11eb5d3557805715077b30aefebc",
        },
        {
            "url": "https://huggingface.co/wok000/weights/resolve/7a376af24f4f21f9d9e24160ed9d858e8a33bf93/crepe/onnx/tiny.onnx",
            "saveTo": os.path.join(pretrained_weights_dir, CREPE_ONNX_TINY),
            "hash": "b509427f6d223152e57ff2aeb1b48300",
        },
        {
            "url": "https://github.com/maxrmorrison/torchcrepe/raw/745670a18bf8c5f1a2f08c910c72433badde3e08/torchcrepe/assets/full.pth",
            "saveTo": os.path.join(pretrained_weights_dir, CREPE_FULL),
            "hash": "2ab425d128692f27ad5b765f13752333",
        },
        {
            "url": "https://github.com/maxrmorrison/torchcrepe/raw/745670a18bf8c5f1a2f08c910c72433badde3e08/torchcrepe/assets/tiny.pth",
            "saveTo": os.path.join(pretrained_weights_dir, CREPE_TINY),
            "hash": "eec11d7661587b6b90da7823cf409340",
        },
        {
            "url": "https://huggingface.co/wok000/weights_gpl/resolve/c2f3e4a8884dba0995347dfe24dc0ad40acb9eb7/content-vec/contentvec-f.onnx",
            "saveTo": os.path.join(pretrained_weights_dir, CONTENT_VEC_500_ONNX),
            "hash": "ab288ca5b540a4a15909a40edf875d1e",
        },
        {
            "url": "https://huggingface.co/wok000/weights/resolve/4a9dbeb086b66721378b4fb29c84bf94d3e076ec/rmvpe/rmvpe_20231006.pt",
            "saveTo": os.path.join(pretrained_weights_dir, RMVPE),
            "hash": "7989809b6b54fb33653818e357bcb643",
        },
        {
            "url": "https://huggingface.co/deiteris/weights/resolve/5040af391eb55d6415a209bfeb3089a866491670/rmvpe_upd.onnx",
            "saveTo": os.path.join(pretrained_weights_dir, RMVPE_ONNX),
            "hash": "9c6d7712f84d487ae781b0d7435c269b",
        },
        {
            "url": "https://github.com/CNChTu/FCPE/raw/819765c8db719c457f53aaee3238879ab98ed0cd/torchfcpe/assets/fcpe_c_v001.pt",
            "saveTo": os.path.join(pretrained_weights_dir, FCPE),
            "hash": "933f1b588409b3945389381a2ab98014",
        },
        {
            "url": "https://huggingface.co/deiteris/weights/resolve/6abbb0285b1fc154e112b3c002ae63e1c1733d53/fcpe.onnx",
            "saveTo": os.path.join(pretrained_weights_dir, FCPE_ONNX),
            "hash": "6a7b11db05def00053102920d039760f",
        },
    ]

    files_to_download = []
    for param in file_params:
        files_to_download.append(
            {
                "url": param["url"],
                "saveTo": param["saveTo"],
                "hash": param["hash"],
            }
        )

    tasks: list[asyncio.Task] = []
    for file in files_to_download:
        tasks.append(asyncio.ensure_future(download(file, hash_cache_file)))
    fail = False
    for i, res in enumerate(await asyncio.gather(*tasks, return_exceptions=True)):
        if isinstance(res, Exception):
            logger.error(
                f'Failed to download or verify {files_to_download[i]["saveTo"]}'
            )
            fail = True
            logger.exception(res)
    if fail:
        raise PretrainDownloadException()

    logger.info("All weights are loaded!")
