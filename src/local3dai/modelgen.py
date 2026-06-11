from __future__ import annotations

import hashlib
import math
import shutil
import subprocess
from pathlib import Path


def generate_3d_model(
    *,
    prompt: str,
    output: str | Path,
    backend: str = "sample",
    external_command: str = "",
) -> Path:
    output_path = Path(output)
    if output_path.suffix.lower() not in {".obj", ".glb", ".gltf", ".fbx"}:
        output_path.mkdir(parents=True, exist_ok=True)
        output_path = output_path / "generated_model.obj"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    normalized = backend.strip().lower()
    if normalized == "sample":
        sample = Path.cwd() / "examples" / "sample_model.obj"
        if not sample.exists():
            sample = Path(__file__).resolve().parents[2] / "examples" / "sample_model.obj"
        shutil.copy2(sample, output_path)
        return output_path

    if normalized == "procedural-crystal":
        _write_procedural_crystal(output_path, prompt)
        return output_path

    if normalized == "external":
        if not external_command:
            raise RuntimeError("model_generation.external_command is empty")
        command = external_command.format(prompt=prompt, output=str(output_path))
        subprocess.run(command, check=True, shell=True)
        if not output_path.exists():
            raise RuntimeError(f"External 3D generator did not create expected model: {output_path}")
        return output_path

    raise ValueError(f"Unknown 3D generation backend: {backend}")


def _write_procedural_crystal(output_path: Path, prompt: str) -> None:
    digest = hashlib.sha256(prompt.encode("utf-8")).digest()
    phase = digest[0] / 255.0 * math.tau
    accent_shift = (digest[1] / 255.0 - 0.5) * 0.35
    meshes: list[tuple[str, str, list[tuple[float, float, float]], list[tuple[int, ...]]]] = []

    def add_mesh(name: str, material: str, vertices: list[tuple[float, float, float]], faces: list[tuple[int, ...]]) -> None:
        meshes.append((name, material, vertices, faces))

    def ring_points(
        sides: int,
        radius_x: float,
        radius_y: float,
        z: float,
        *,
        center: tuple[float, float] = (0.0, 0.0),
        twist: float = 0.0,
    ) -> list[tuple[float, float, float]]:
        cx, cy = center
        return [
            (
                cx + math.cos(twist + math.tau * i / sides) * radius_x,
                cy + math.sin(twist + math.tau * i / sides) * radius_y,
                z,
            )
            for i in range(sides)
        ]

    def frustum(name: str, material: str, sides: int, z0: float, z1: float, r0: float, r1: float, *, twist: float = 0.0) -> None:
        vertices = ring_points(sides, r0, r0, z0, twist=twist) + ring_points(sides, r1, r1, z1, twist=twist + math.pi / sides)
        faces: list[tuple[int, ...]] = []
        for i in range(sides):
            faces.append((i + 1, (i + 1) % sides + 1, sides + (i + 1) % sides + 1, sides + i + 1))
        faces.append(tuple(range(sides, 0, -1)))
        faces.append(tuple(range(sides + 1, sides * 2 + 1)))
        add_mesh(name, material, vertices, faces)

    def obelisk(name: str, material: str) -> None:
        sides = 10
        lower = ring_points(sides, 0.46, 0.46, 0.22, twist=phase)
        shoulder = ring_points(sides, 0.34 + accent_shift * 0.08, 0.34, 2.05, twist=phase + 0.24)
        crown = ring_points(sides, 0.18, 0.18, 2.46, twist=phase + 0.51)
        vertices = lower + shoulder + crown + [(0.0, 0.0, 3.18)]
        tip = len(vertices)
        faces: list[tuple[int, ...]] = []
        for i in range(sides):
            faces.append((i + 1, (i + 1) % sides + 1, sides + (i + 1) % sides + 1, sides + i + 1))
            faces.append((sides + i + 1, sides + (i + 1) % sides + 1, sides * 2 + (i + 1) % sides + 1, sides * 2 + i + 1))
            faces.append((sides * 2 + i + 1, sides * 2 + (i + 1) % sides + 1, tip))
        faces.append(tuple(range(sides, 0, -1)))
        add_mesh(name, material, vertices, faces)

    def bipyramid(name: str, material: str, sides: int, center: tuple[float, float, float], radius: float, height: float, *, twist: float) -> None:
        cx, cy, cz = center
        ring = ring_points(sides, radius * 0.78, radius, cz, center=(cx, cy), twist=twist)
        vertices = [(cx, cy, cz - height * 0.48)] + ring + [(cx, cy, cz + height * 0.52)]
        top = len(vertices)
        faces: list[tuple[int, ...]] = []
        for i in range(sides):
            faces.append((1, i + 2, (i + 1) % sides + 2))
            faces.append((top, (i + 1) % sides + 2, i + 2))
        add_mesh(name, material, vertices, faces)

    def vertical_ring(name: str, material: str, segments: int, radius: float, width: float, depth: float, zc: float) -> None:
        vertices: list[tuple[float, float, float]] = []
        for i in range(segments):
            angle = math.tau * i / segments + phase * 0.15
            for r, y in ((radius + width, -depth), (radius - width, -depth), (radius + width, depth), (radius - width, depth)):
                vertices.append((math.cos(angle) * r, y, zc + math.sin(angle) * r))
        faces: list[tuple[int, ...]] = []
        for i in range(segments):
            nxt = (i + 1) % segments
            a, b = i * 4 + 1, nxt * 4 + 1
            faces.append((a, b, b + 2, a + 2))
            faces.append((a + 1, a + 3, b + 3, b + 1))
            faces.append((a, a + 1, b + 1, b))
            faces.append((a + 2, b + 2, b + 3, a + 3))
        add_mesh(name, material, vertices, faces)

    frustum("octagonal_obsidian_plinth", "obsidian", 8, -0.08, 0.18, 1.06, 0.86, twist=phase * 0.2)
    frustum("cyan_inner_altar", "cyan_glow", 12, 0.16, 0.30, 0.56, 0.42, twist=phase * 0.4)
    vertical_ring("tilted_energy_halo", "magenta_glow", 28, 1.03, 0.045, 0.035, 1.33)
    obelisk("faceted_neon_core", "ice_crystal")
    for i in range(8):
        angle = phase + math.tau * i / 8
        radius = 0.86 + (digest[i + 2] / 255.0) * 0.18
        x = math.cos(angle) * radius
        y = math.sin(angle) * radius
        height = 0.55 + (digest[i + 10] / 255.0) * 0.45
        z = 0.45 + (i % 3) * 0.11
        material = "cyan_glow" if i % 2 == 0 else "magenta_glow"
        bipyramid(f"orbiting_shard_{i:02d}", material, 5 + i % 3, (x, y, z), 0.12 + 0.02 * (i % 2), height, twist=angle)

    _write_mtl(output_path.with_suffix(".mtl"))
    with output_path.open("w", encoding="utf-8") as fh:
        fh.write(f"mtllib {output_path.with_suffix('.mtl').name}\n")
        offset = 0
        for name, material, vertices, faces in meshes:
            fh.write(f"\no {name}\nusemtl {material}\n")
            for x, y, z in vertices:
                fh.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")
            for face in faces:
                indices = " ".join(str(offset + idx) for idx in face)
                fh.write(f"f {indices}\n")
            offset += len(vertices)


def _write_mtl(path: Path) -> None:
    path.write_text(
        """newmtl obsidian
Ka 0.015 0.018 0.026
Kd 0.030 0.035 0.055
Ks 0.800 0.820 0.900
Ns 420

newmtl ice_crystal
Ka 0.210 0.260 0.320
Kd 0.540 0.720 0.860
Ks 0.950 0.960 1.000
Ns 520
d 0.92

newmtl cyan_glow
Ka 0.000 0.360 0.500
Kd 0.000 0.900 1.000
Ks 0.800 1.000 1.000
Ns 360

newmtl magenta_glow
Ka 0.420 0.020 0.520
Kd 1.000 0.080 0.850
Ks 1.000 0.780 1.000
Ns 360
""",
        encoding="utf-8",
    )
