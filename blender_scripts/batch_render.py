from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

import bpy
from mathutils import Vector


def parse_args() -> argparse.Namespace:
    import sys

    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--views", type=int, default=8)
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--engine", default="CYCLES")
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--camera-distance", type=float, default=3.2)
    return parser.parse_args(argv)


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def import_model(path: Path) -> None:
    suffix = path.suffix.lower()
    if suffix in {".glb", ".gltf"}:
        bpy.ops.import_scene.gltf(filepath=str(path))
    elif suffix == ".obj":
        try:
            bpy.ops.wm.obj_import(filepath=str(path))
        except Exception:
            bpy.ops.import_scene.obj(filepath=str(path))
    elif suffix == ".fbx":
        bpy.ops.import_scene.fbx(filepath=str(path))
    else:
        raise ValueError(f"Unsupported model format: {path.suffix}")


def mesh_objects() -> list[bpy.types.Object]:
    return [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]


def normalize_model() -> None:
    objects = mesh_objects()
    if not objects:
        raise RuntimeError("No mesh objects were imported")
    min_corner = Vector((float("inf"), float("inf"), float("inf")))
    max_corner = Vector((float("-inf"), float("-inf"), float("-inf")))
    for obj in objects:
        for corner in obj.bound_box:
            world = obj.matrix_world @ Vector(corner)
            min_corner.x = min(min_corner.x, world.x)
            min_corner.y = min(min_corner.y, world.y)
            min_corner.z = min(min_corner.z, world.z)
            max_corner.x = max(max_corner.x, world.x)
            max_corner.y = max(max_corner.y, world.y)
            max_corner.z = max(max_corner.z, world.z)
    center = (min_corner + max_corner) / 2
    size = max((max_corner - min_corner).x, (max_corner - min_corner).y, (max_corner - min_corner).z)
    scale = 1.8 / size if size else 1.0
    for obj in objects:
        obj.location -= center
        obj.scale *= scale
    bpy.context.view_layer.update()


def configure_scene(args: argparse.Namespace) -> None:
    scene = bpy.context.scene
    try:
        scene.render.engine = args.engine
    except TypeError:
        scene.render.engine = "CYCLES"
    if scene.render.engine == "CYCLES":
        scene.cycles.samples = args.samples
        scene.cycles.use_denoising = True
    scene.render.resolution_x = args.resolution
    scene.render.resolution_y = args.resolution
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.film_transparent = False
    try:
        scene.view_settings.view_transform = "Filmic"
        scene.view_settings.look = "Medium High Contrast"
    except TypeError:
        scene.view_settings.view_transform = "Standard"
    scene.world = bpy.data.worlds.new("World") if scene.world is None else scene.world
    scene.world.color = (0.94, 0.95, 0.97)


def make_camera(distance: float) -> bpy.types.Object:
    bpy.ops.object.camera_add(location=(0, -distance, distance * 0.42), rotation=(math.radians(66), 0, 0))
    camera = bpy.context.object
    camera.data.lens = 70
    camera.data.type = "ORTHO"
    camera.data.ortho_scale = 2.7
    bpy.context.scene.camera = camera
    return camera


def point_camera(camera: bpy.types.Object, target: Vector = Vector((0, 0, 0.05))) -> None:
    direction = target - camera.location
    camera.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def add_lights() -> None:
    bpy.ops.object.light_add(type="AREA", location=(-3, -4, 5))
    key = bpy.context.object
    key.data.energy = 550
    key.data.size = 4
    bpy.ops.object.light_add(type="AREA", location=(3.5, 2.5, 3.4))
    fill = bpy.context.object
    fill.data.energy = 110
    fill.data.size = 5


def assign_emission(name: str, color: tuple[float, float, float, float]) -> None:
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    nodes.clear()
    out = nodes.new(type="ShaderNodeOutputMaterial")
    emission = nodes.new(type="ShaderNodeEmission")
    emission.inputs["Color"].default_value = color
    emission.inputs["Strength"].default_value = 1.0
    material.node_tree.links.new(emission.outputs["Emission"], out.inputs["Surface"])
    for obj in mesh_objects():
        obj.data.materials.clear()
        obj.data.materials.append(material)


def capture_materials() -> dict[str, list[bpy.types.Material]]:
    return {obj.name: [slot.material for slot in obj.material_slots] for obj in mesh_objects()}


def restore_materials(snapshot: dict[str, list[bpy.types.Material]]) -> None:
    for obj in mesh_objects():
        obj.data.materials.clear()
        for material in snapshot.get(obj.name, []):
            if material is not None:
                obj.data.materials.append(material)


def assign_normal_material() -> None:
    material = bpy.data.materials.new("normal_pass_material")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    nodes.clear()
    links = material.node_tree.links
    out = nodes.new(type="ShaderNodeOutputMaterial")
    geom = nodes.new(type="ShaderNodeNewGeometry")
    add = nodes.new(type="ShaderNodeVectorMath")
    add.operation = "ADD"
    add.inputs[1].default_value = (1, 1, 1)
    multiply = nodes.new(type="ShaderNodeVectorMath")
    multiply.operation = "MULTIPLY"
    multiply.inputs[1].default_value = (0.5, 0.5, 0.5)
    emission = nodes.new(type="ShaderNodeEmission")
    emission.inputs["Strength"].default_value = 1.0
    links.new(geom.outputs["Normal"], add.inputs[0])
    links.new(add.outputs["Vector"], multiply.inputs[0])
    links.new(multiply.outputs["Vector"], emission.inputs["Color"])
    links.new(emission.outputs["Emission"], out.inputs["Surface"])
    for obj in mesh_objects():
        obj.data.materials.clear()
        obj.data.materials.append(material)


def assign_depth_material(max_distance: float) -> None:
    material = bpy.data.materials.new("depth_pass_material")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    nodes.clear()
    links = material.node_tree.links
    out = nodes.new(type="ShaderNodeOutputMaterial")
    camera_data = nodes.new(type="ShaderNodeCameraData")
    map_range = nodes.new(type="ShaderNodeMapRange")
    map_range.inputs["From Min"].default_value = 0.0
    map_range.inputs["From Max"].default_value = max(0.1, max_distance)
    map_range.inputs["To Min"].default_value = 1.0
    map_range.inputs["To Max"].default_value = 0.05
    map_range.clamp = True
    emission = nodes.new(type="ShaderNodeEmission")
    emission.inputs["Strength"].default_value = 1.0
    links.new(camera_data.outputs["View Distance"], map_range.inputs["Value"])
    links.new(map_range.outputs["Result"], emission.inputs["Color"])
    links.new(emission.outputs["Emission"], out.inputs["Surface"])
    for obj in mesh_objects():
        obj.data.materials.clear()
        obj.data.materials.append(material)


def save_render(path: Path) -> None:
    bpy.context.scene.render.filepath = str(path)
    bpy.ops.render.render(write_still=True)


def ensure_freestyle_line_set() -> None:
    view_layer = bpy.context.scene.view_layers[0]
    if not view_layer.freestyle_settings.linesets:
        view_layer.freestyle_settings.linesets.new("LineSet")
    view_layer.freestyle_settings.linesets[0].linestyle.thickness = 2.2


def render_view(
    view_id: str,
    output: Path,
    camera: bpy.types.Object,
    angle: float,
    orbit_radius: float,
    material_snapshot: dict[str, list[bpy.types.Material]],
) -> dict[str, str]:
    view_dir = output / view_id
    view_dir.mkdir(parents=True, exist_ok=True)
    restore_materials(material_snapshot)
    camera.location = (math.sin(angle) * orbit_radius, -math.cos(angle) * orbit_radius, camera.location.z)
    point_camera(camera)

    scene = bpy.context.scene
    original_world = scene.world.color[:]
    scene.render.use_freestyle = False
    rgb_path = view_dir / "rgb.png"
    save_render(rgb_path)

    scene.world.color = (0, 0, 0)
    assign_depth_material(orbit_radius * 2.5)
    depth_path = view_dir / "depth.png"
    save_render(depth_path)

    scene.world.color = (0, 0, 0)
    assign_emission("mask_white", (1, 1, 1, 1))
    mask_path = view_dir / "mask.png"
    save_render(mask_path)

    scene.render.use_freestyle = True
    ensure_freestyle_line_set()
    edge_path = view_dir / "edge.png"
    save_render(edge_path)
    scene.render.use_freestyle = False

    assign_normal_material()
    normal_path = view_dir / "normal.png"
    save_render(normal_path)
    scene.world.color = original_world
    return {
        "rgb": str(rgb_path),
        "depth": str(depth_path),
        "edge": str(edge_path),
        "normal": str(normal_path),
        "mask": str(mask_path),
    }


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    clear_scene()
    import_model(Path(args.model))
    normalize_model()
    configure_scene(args)
    add_lights()
    camera = make_camera(args.camera_distance)
    material_snapshot = capture_materials()

    manifest = {"type": "render_manifest", "source": str(args.model), "views": []}
    for index in range(args.views):
        angle = math.tau * index / max(args.views, 1)
        view_id = f"view_{index:02d}"
        files = render_view(view_id, output, camera, angle, args.camera_distance, material_snapshot)
        manifest["views"].append({"view_id": view_id, "azimuth_deg": round(math.degrees(angle), 3), "files": files})

    with (output / "manifest.json").open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)
        fh.write("\n")

    # Blender's compositor can leave numbered temporary files around if a render is interrupted.
    for tmp in output.glob("**/*0001.png"):
        target = tmp.with_name(tmp.name.replace("0001", ""))
        if not target.exists():
            shutil.move(str(tmp), str(target))


if __name__ == "__main__":
    main()
