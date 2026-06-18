from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

import bpy
from mathutils import Matrix, Vector


def parse_args() -> argparse.Namespace:
    import sys

    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--views", type=int, default=8)
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--resolution-x", type=int, default=0)
    parser.add_argument("--resolution-y", type=int, default=0)
    parser.add_argument("--engine", default="CYCLES")
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--camera-distance", type=float, default=3.2)
    parser.add_argument("--camera-json", default="", help="Optional fixed CameraState JSON or path. Renders one view_locked view.")
    parser.add_argument("--preview-only", action="store_true", help="Render RGB previews only. Used for camera candidate selection.")
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
    transform = Matrix.Diagonal((scale, scale, scale, 1.0)) @ Matrix.Translation(-center)
    for obj in objects:
        obj.matrix_world = transform @ obj.matrix_world
    bpy.context.view_layer.update()


def configure_scene(args: argparse.Namespace) -> None:
    scene = bpy.context.scene
    engine_candidates = [args.engine]
    if args.engine == "BLENDER_EEVEE_NEXT":
        engine_candidates.extend(["BLENDER_EEVEE", "CYCLES"])
    elif args.engine == "BLENDER_EEVEE":
        engine_candidates.extend(["BLENDER_EEVEE_NEXT", "CYCLES"])
    else:
        engine_candidates.append("CYCLES")
    for engine in engine_candidates:
        try:
            scene.render.engine = engine
            break
        except TypeError:
            continue
    if scene.render.engine == "CYCLES":
        scene.cycles.samples = args.samples
        scene.cycles.use_denoising = True
    scene.render.resolution_x = args.resolution_x or args.resolution
    scene.render.resolution_y = args.resolution_y or args.resolution
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


def apply_camera_projection(camera: bpy.types.Object, camera_state: dict[str, object]) -> None:
    camera_type = str(camera_state.get("camera_type", camera_state.get("type", "orthographic"))).lower()
    camera.data.shift_x = float(camera_state.get("shift_x", 0.0) or 0.0)
    camera.data.shift_y = float(camera_state.get("shift_y", 0.0) or 0.0)
    if camera_type in {"perspective", "persp"}:
        camera.data.type = "PERSP"
        camera.data.lens = max(12.0, float(camera_state.get("focal_length_mm", camera_state.get("lens", 58.0))))
        camera.data.sensor_width = max(12.0, float(camera_state.get("sensor_width_mm", 36.0)))
    else:
        camera.data.type = "ORTHO"
        camera.data.ortho_scale = max(0.4, float(camera_state.get("ortho_scale", camera.data.ortho_scale)))


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


def assign_clay_material() -> None:
    material = bpy.data.materials.new("white_clay_preview_material")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    principled = nodes.get("Principled BSDF")
    if principled:
        principled.inputs["Base Color"].default_value = (0.86, 0.88, 0.9, 1)
        principled.inputs["Roughness"].default_value = 0.52
        principled.inputs["Metallic"].default_value = 0.0
    material.diffuse_color = (0.86, 0.88, 0.9, 1)
    for obj in mesh_objects():
        obj.data.materials.clear()
        obj.data.materials.append(material)


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


def camera_position_from_state(camera_state: dict[str, object], base_distance: float) -> Vector:
    position_values = camera_state.get("position")
    if isinstance(position_values, list | tuple) and len(position_values) == 3:
        return Vector((float(position_values[0]), float(position_values[1]), float(position_values[2])))
    target_values = camera_state.get("target", [0.0, 0.0, 0.05])
    if not isinstance(target_values, list | tuple) or len(target_values) != 3:
        target_values = [0.0, 0.0, 0.05]
    target = Vector((float(target_values[0]), float(target_values[1]), float(target_values[2])))
    azimuth = math.radians(float(camera_state.get("azimuth_deg", 35.0)))
    elevation = math.radians(float(camera_state.get("elevation_deg", 18.0)))
    distance_scale = max(0.25, float(camera_state.get("distance_scale", 1.0)))
    distance = base_distance * distance_scale
    horizontal = math.cos(elevation) * distance
    return Vector(
        (
            target.x + math.sin(azimuth) * horizontal,
            target.y - math.cos(azimuth) * horizontal,
            target.z + math.sin(elevation) * distance,
        )
    )


def load_camera_state(raw: str) -> dict[str, object] | None:
    if not raw:
        return None
    stripped = raw.strip()
    if stripped.startswith("{"):
        return json.loads(stripped)
    path = Path(raw)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return json.loads(stripped)


def render_channels(
    view_id: str,
    output: Path,
    camera: bpy.types.Object,
    max_depth_distance: float,
    material_snapshot: dict[str, list[bpy.types.Material]],
) -> dict[str, str]:
    view_dir = output / view_id
    view_dir.mkdir(parents=True, exist_ok=True)
    restore_materials(material_snapshot)

    scene = bpy.context.scene
    original_world = scene.world.color[:]
    scene.render.use_freestyle = False
    rgb_path = view_dir / "rgb.png"
    save_render(rgb_path)

    scene.world.color = (0, 0, 0)
    assign_depth_material(max_depth_distance)
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


def render_view(
    view_id: str,
    output: Path,
    camera: bpy.types.Object,
    angle: float,
    orbit_radius: float,
    material_snapshot: dict[str, list[bpy.types.Material]],
) -> dict[str, str]:
    camera.location = (math.sin(angle) * orbit_radius, -math.cos(angle) * orbit_radius, camera.location.z)
    point_camera(camera)
    return render_channels(view_id, output, camera, orbit_radius * 2.5, material_snapshot)


def render_locked_view(
    output: Path,
    camera: bpy.types.Object,
    camera_state: dict[str, object],
    base_distance: float,
    material_snapshot: dict[str, list[bpy.types.Material]],
    *,
    view_id: str = "view_locked",
    yaw_offset_deg: float = 0.0,
) -> dict[str, str]:
    target_values = camera_state.get("target", [0.0, 0.0, 0.05])
    if not isinstance(target_values, list | tuple) or len(target_values) != 3:
        target_values = [0.0, 0.0, 0.05]
    target = Vector((float(target_values[0]), float(target_values[1]), float(target_values[2])))
    base_location = camera_position_from_state(camera_state, base_distance)
    if yaw_offset_deg:
        rel = base_location - target
        angle = math.radians(yaw_offset_deg)
        camera.location = target + Vector(
            (
                rel.x * math.cos(angle) - rel.y * math.sin(angle),
                rel.x * math.sin(angle) + rel.y * math.cos(angle),
                rel.z,
            )
        )
    else:
        camera.location = base_location
    apply_camera_projection(camera, camera_state)
    point_camera(camera, target)
    return render_channels(view_id, output, camera, base_distance * max(0.25, float(camera_state.get("distance_scale", 1.0))) * 2.5, material_snapshot)


def render_locked_rgb_preview(
    output: Path,
    camera: bpy.types.Object,
    camera_state: dict[str, object],
    base_distance: float,
    material_snapshot: dict[str, list[bpy.types.Material]],
    *,
    view_id: str,
) -> dict[str, str]:
    target_values = camera_state.get("target", [0.0, 0.0, 0.05])
    if not isinstance(target_values, list | tuple) or len(target_values) != 3:
        target_values = [0.0, 0.0, 0.05]
    target = Vector((float(target_values[0]), float(target_values[1]), float(target_values[2])))
    camera.location = camera_position_from_state(camera_state, base_distance)
    apply_camera_projection(camera, camera_state)
    point_camera(camera, target)
    restore_materials(material_snapshot)
    bpy.context.scene.render.use_freestyle = False
    view_dir = output / view_id
    view_dir.mkdir(parents=True, exist_ok=True)
    rgb_path = view_dir / "rgb.png"
    save_render(rgb_path)
    return {"rgb": str(rgb_path)}


def camera_candidates_from_payload(camera_payload: dict[str, object]) -> list[dict[str, object]]:
    raw_candidates = camera_payload.get("candidate_views")
    if raw_candidates is None and camera_payload.get("mode") == "camera_search":
        raw_candidates = camera_payload.get("views")
    if not isinstance(raw_candidates, list):
        return []
    candidates: list[dict[str, object]] = []
    for index, item in enumerate(raw_candidates):
        if not isinstance(item, dict):
            continue
        candidate = dict(item)
        candidate.setdefault("view_id", f"camera_candidate_{index:02d}")
        candidates.append(candidate)
    return candidates


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    clear_scene()
    import_model(Path(args.model))
    normalize_model()
    configure_scene(args)
    add_lights()
    assign_clay_material()
    camera = make_camera(args.camera_distance)
    material_snapshot = capture_materials()

    manifest = {"type": "render_manifest", "source": str(args.model), "views": []}
    fixed_camera = load_camera_state(args.camera_json)
    if fixed_camera:
        camera_candidates = camera_candidates_from_payload(fixed_camera)
        if camera_candidates:
            manifest["type"] = "camera_preview_manifest"
            manifest["source"] = str(args.model)
            for candidate in camera_candidates:
                view_id = str(candidate.get("view_id", f"camera_candidate_{len(manifest['views']):02d}"))
                files = render_locked_rgb_preview(
                    output,
                    camera,
                    candidate,
                    args.camera_distance,
                    material_snapshot,
                    view_id=view_id,
                )
                manifest["views"].append({"view_id": view_id, "camera": candidate, "files": files})
        elif args.preview_only:
            manifest["type"] = "camera_preview_manifest"
            files = render_locked_rgb_preview(
                output,
                camera,
                fixed_camera,
                args.camera_distance,
                material_snapshot,
                view_id="view_locked",
            )
            manifest["views"].append({"view_id": "view_locked", "camera": fixed_camera, "files": files})
        else:
            locked_specs = [
                ("view_locked", 0.0),
                ("view_left_30", 30.0),
                ("view_right_30", -30.0),
            ]
            for view_id, yaw_offset in locked_specs:
                files = render_locked_view(
                    output,
                    camera,
                    fixed_camera,
                    args.camera_distance,
                    material_snapshot,
                    view_id=view_id,
                    yaw_offset_deg=yaw_offset,
                )
                manifest["views"].append(
                    {
                        "view_id": view_id,
                        "camera": fixed_camera,
                        "yaw_offset_deg": yaw_offset,
                        "files": files,
                    }
                )
            manifest["view_graph"] = {
                "edges": [
                    {"source": "view_locked", "target": "view_left_30", "overlap_ratio": 0.66},
                    {"source": "view_locked", "target": "view_right_30", "overlap_ratio": 0.66},
                    {"source": "view_left_30", "target": "view_right_30", "overlap_ratio": 0.42},
                ]
            }
    else:
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
