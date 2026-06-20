from __future__ import annotations

import aviary
import aviary.tile
import pydantic
from ultralytics import YOLO


class VehicleModelConfig(pydantic.BaseModel):
    r_channel_name: aviary.ChannelName | str = aviary.ChannelName.R
    g_channel_name: aviary.ChannelName | str = aviary.ChannelName.G
    b_channel_name: aviary.ChannelName | str = aviary.ChannelName.B
    vehicle_channel_name: str = 'vehicle'
    remove_channels: bool = True


@aviary.tile.register_tiles_processor(
    config_class=VehicleModelConfig,
)
@aviary.log
class VehicleModel(aviary.IDMixin):

    def __init__(
        self,
        r_channel_name: aviary.ChannelName | str = aviary.ChannelName.R,
        g_channel_name: aviary.ChannelName | str = aviary.ChannelName.G,
        b_channel_name: aviary.ChannelName | str = aviary.ChannelName.B,
        vehicle_channel_name: str = 'vehicle',
        remove_channels: bool = True,
    ) -> None:
        self._r_channel_name = r_channel_name
        self._g_channel_name = g_channel_name
        self._b_channel_name = b_channel_name
        self._vehicle_channel_name = vehicle_channel_name
        self._remove_channels = remove_channels

        self._model = YOLO('yolo26l-obb.pt')

        super().__init__()

    @classmethod
    def from_config(
        cls,
        config: VehicleModelConfig,
    ) -> VehicleModel:
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
        inputs_list = list(inputs)

        results = self._model.predict(inputs_list, verbose=False)

        batched_objects: list[list[aviary.Object]] = []

        for img, res in zip(inputs_list, results, strict=True):
            height, width = img.shape[0], img.shape[1]

            objects: list[aviary.Object] = []

            obb = getattr(res, 'obb', None)
            if obb is None or getattr(obb, 'xywhr', None) is None or len(obb) == 0:
                batched_objects.append(objects)
                continue

            xywhr = obb.xywhr.tolist()
            clss = obb.cls.tolist()
            confs = obb.conf.tolist()

            for (xc, yc, w, h, r), cls_val, conf in zip(xywhr, clss, confs, strict=True):
                x_center = float(xc) / float(width) if width else 0.
                y_center_img = float(yc) / float(height) if height else 0.
                y_center = 1. - y_center_img
                norm_w = float(w) / float(width) if width else 0.
                norm_h = float(h) / float(height) if height else 0.

                rotation = float(-r)

                value = int(cls_val) if isinstance(cls_val, (int, float)) else cls_val

                objects.append(
                    aviary.Object(
                        value=value,
                        x_center=float(x_center),
                        y_center=float(y_center),
                        width=float(norm_w),
                        height=float(norm_h),
                        rotation=rotation,
                        score=float(conf),
                    )
                )

            batched_objects.append(objects)

        buffer_fraction = 0.
        for channel_name in (self._r_channel_name, self._g_channel_name, self._b_channel_name):
            if channel_name in tiles:
                buffer_fraction = tiles[channel_name].buffer_size
                break

        vehicle_channel = aviary.ObjectChannel(
            data=batched_objects,
            name=self._vehicle_channel_name,
            buffer_size=buffer_fraction,
            copy=False,
        )

        tiles = tiles.append(
            channels=vehicle_channel,
            inplace=True,
        )

        if self._remove_channels:
            channel_names = set(channel_names)
            tiles = tiles.remove(
                channel_names=channel_names,
                inplace=True,
            )

        return tiles
