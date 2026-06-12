from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any


Vector3 = tuple[float, float, float]


@dataclass
class CameraState:
    azimuth_deg: float = 35.0
    elevation_deg: float = 18.0
    distance_scale: float = 1.0
    ortho_scale: float = 2.7
    target: Vector3 = (0.0, 0.0, 0.05)
    position: Vector3 | None = None
    viewport_aspect: float = 1.0
    coordinate_space: str = "blender_z_up"

    @classmethod
    def from_payload(cls, payload: dict[str, Any] | None) -> "CameraState":
        data = payload or {}
        target = data.get("target", cls.target)
        if not isinstance(target, (list, tuple)) or len(target) != 3:
            target = cls.target
        position = data.get("position")
        if not isinstance(position, (list, tuple)) or len(position) != 3:
            position = None
        return cls(
            azimuth_deg=float(data.get("azimuth_deg", cls.azimuth_deg)),
            elevation_deg=float(data.get("elevation_deg", cls.elevation_deg)),
            distance_scale=max(0.25, float(data.get("distance_scale", cls.distance_scale))),
            ortho_scale=max(0.4, float(data.get("ortho_scale", cls.ortho_scale))),
            target=(float(target[0]), float(target[1]), float(target[2])),
            position=None if position is None else (float(position[0]), float(position[1]), float(position[2])),
            viewport_aspect=max(0.2, min(4.0, float(data.get("viewport_aspect", cls.viewport_aspect)))),
            coordinate_space=str(data.get("coordinate_space", cls.coordinate_space)),
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["target"] = list(self.target)
        if self.position is not None:
            data["position"] = list(self.position)
        return data

    def to_blender(self, *, model_path: str | None = None) -> "CameraState":
        """Convert browser camera state into Blender's normalized Z-up space."""
        suffix = (model_path or "").lower()
        if self.coordinate_space == "three_y_up":
            target = _three_y_up_to_blender_z_up(self.target)
            position = None if self.position is None else _three_y_up_to_blender_z_up(self.position)
            return CameraState(
                azimuth_deg=self.azimuth_deg,
                elevation_deg=self.elevation_deg,
                distance_scale=self.distance_scale,
                ortho_scale=self.ortho_scale,
                target=target,
                position=position,
                viewport_aspect=self.viewport_aspect,
                coordinate_space="blender_z_up",
            )
        if self.coordinate_space == "obj_z_up" or suffix.endswith(".obj"):
            return CameraState(
                azimuth_deg=self.azimuth_deg,
                elevation_deg=self.elevation_deg,
                distance_scale=self.distance_scale,
                ortho_scale=self.ortho_scale,
                target=self.target,
                position=self.position,
                viewport_aspect=self.viewport_aspect,
                coordinate_space="blender_z_up",
            )
        return CameraState(
            azimuth_deg=self.azimuth_deg,
            elevation_deg=self.elevation_deg,
            distance_scale=self.distance_scale,
            ortho_scale=self.ortho_scale,
            target=self.target,
            position=self.position,
            viewport_aspect=self.viewport_aspect,
            coordinate_space="blender_z_up",
        )


def camera_position(state: CameraState, *, base_distance: float = 3.2) -> tuple[float, float, float]:
    if state.position is not None:
        return state.position
    distance = max(0.1, base_distance * state.distance_scale)
    azimuth = math.radians(state.azimuth_deg)
    elevation = math.radians(state.elevation_deg)
    horizontal = math.cos(elevation) * distance
    x = state.target[0] + math.sin(azimuth) * horizontal
    y = state.target[1] - math.cos(azimuth) * horizontal
    z = state.target[2] + math.sin(elevation) * distance
    return (x, y, z)


def _three_y_up_to_blender_z_up(value: tuple[float, float, float]) -> tuple[float, float, float]:
    x, y, z = value
    return (x, -z, y)
