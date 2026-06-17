from __future__ import annotations

from enum import Enum
from pathlib import Path

import aviary
import aviary.tile
import numpy as np
import onnxruntime
import pydantic
from huggingface_hub import hf_hub_download


class Device(Enum):
    CPU = 'cpu'
    CUDA = 'cuda'
    TENSORRT = 'tensorrt'

    def to_onnx(self) -> str:
        mapping = {
            Device.CPU: 'CPUExecutionProvider',
            Device.CUDA: 'CUDAExecutionProvider',
            Device.TENSORRT: 'TensorrtExecutionProvider',
        }
        return mapping[self]


class LandCoverModelConfig(pydantic.BaseModel):
    r_channel_name: aviary.ChannelName | str = aviary.ChannelName.R
    g_channel_name: aviary.ChannelName | str = aviary.ChannelName.G
    b_channel_name: aviary.ChannelName | str = aviary.ChannelName.B
    land_cover_channel_name: str = 'land_cover'
    device: Device = Device.CPU
    cache_dir_path: Path = Path('cache')
    remove_channels: bool = True


@aviary.tile.register_tiles_processor(
    config_class=LandCoverModelConfig,
)
@aviary.log
class LandCoverModel(aviary.IDMixin):
    _HF_HUB_REPO = 'geospaitial-lab/FLAIR-HUB_LC-A_RGB_swinbase-upernet_700px'
    _HF_HUB_MODEL_PATH = 'FLAIR-HUB_LC-A_RGB_swinbase-upernet_700px.onnx'

    def __init__(
        self,
        r_channel_name: aviary.ChannelName | str = aviary.ChannelName.R,
        g_channel_name: aviary.ChannelName | str = aviary.ChannelName.G,
        b_channel_name: aviary.ChannelName | str = aviary.ChannelName.B,
        land_cover_channel_name: str = 'land_cover',
        device: Device = Device.CPU,
        cache_dir_path: Path = Path('cache'),
        remove_channels: bool = True,
    ) -> None:
        self._r_channel_name = r_channel_name
        self._g_channel_name = g_channel_name
        self._b_channel_name = b_channel_name
        self._land_cover_channel_name = land_cover_channel_name
        self._device = device.to_onnx()
        self._cache_dir_path = cache_dir_path
        self._remove_channels = remove_channels

        self._model_path = hf_hub_download(
            repo_id=self._HF_HUB_REPO,
            filename=self._HF_HUB_MODEL_PATH,
            local_dir=self._cache_dir_path,
        )
        self._model = onnxruntime.InferenceSession(self._model_path, providers=[self._device])

        super().__init__()

    @classmethod
    def from_config(
        cls,
        config: LandCoverModelConfig,
    ) -> LandCoverModel:
        config = config.model_dump()
        return cls(**config)

    def __call__(
        self,
        tiles: aviary.Tiles,
    ) -> aviary.Tiles:
        channel_names = [
            self._r_channel_name,
            self._g_channel_name,
            self._b_channel_name,
        ]

        inputs = tiles.to_composite_raster(channel_names=channel_names)
        inputs = np.transpose(inputs, axes=(0, 3, 1, 2))

        outputs = self._model.run(
            output_names=None,
            input_feed={'rgb': inputs},
        )

        outputs = np.argmax(outputs[0], axis=1).astype(np.uint8)

        data = list(outputs)
        buffer_size = tiles[channel_names[0]].buffer_size
        land_cover_channel = aviary.RasterChannel(
            data=data,
            name=self._land_cover_channel_name,
            buffer_size=buffer_size,
            copy=False,
        )

        tiles = tiles.append(
            channels=land_cover_channel,
            inplace=True,
        )

        if self._remove_channels:
            channel_names = set(channel_names)
            tiles = tiles.remove(
                channel_names=channel_names,
                inplace=True,
            )

        return tiles
