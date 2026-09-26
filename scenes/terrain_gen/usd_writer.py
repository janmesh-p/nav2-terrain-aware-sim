"""Write generated terrain into a USD stage with PhysX colliders."""

from __future__ import annotations

from pxr import Gf, Sdf, UsdGeom, UsdPhysics, UsdShade, Vt

from .mesh import heightfield_mesh


def _preview_material(stage, path: str, color, roughness: float) -> UsdShade.Material:
    mat = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(roughness))
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return mat


def _physics_material(stage, path: str, spec: dict) -> UsdShade.Material:
    mat = UsdShade.Material.Define(stage, path)
    api = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    api.CreateStaticFrictionAttr(float(spec["static_friction"]))
    api.CreateDynamicFrictionAttr(float(spec["dynamic_friction"]))
    api.CreateRestitutionAttr(float(spec.get("restitution", 0.0)))
    return mat


def write_terrain(stage, cfg: dict, result) -> list[str]:
    """Replace cfg['root_prim'] with freshly generated terrain meshes."""
    root = cfg["root_prim"]
    if stage.GetPrimAtPath(root):
        stage.RemovePrim(root)
    UsdGeom.Xform.Define(stage, root)
    looks = f"{root}/Looks"
    UsdGeom.Scope.Define(stage, looks)

    materials = {}
    for name, spec in cfg["materials"].items():
        visual = _preview_material(
            stage, f"{looks}/{name}_visual", spec["color"], spec.get("render_roughness", 0.85)
        )
        physics = _physics_material(stage, f"{looks}/{name}_physics", spec)
        materials[name] = (visual, physics)

    z_offset = float(cfg.get("z_offset", 0.002))
    written = []
    for p in result.patches:
        path = f"{root}/{p.name}"
        mesh = UsdGeom.Mesh.Define(stage, path)
        pts, counts, indices = heightfield_mesh(p.height, p.x, p.y)
        mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(pts))
        mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(counts))
        mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(indices))
        mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        mesh.CreateExtentAttr(
            Vt.Vec3fArray([Gf.Vec3f(*map(float, lo)), Gf.Vec3f(*map(float, hi))])
        )

        xf = UsdGeom.XformCommonAPI(mesh)
        xf.SetTranslate(Gf.Vec3d(float(p.pose["x"]), float(p.pose["y"]), z_offset))
        xf.SetRotate(Gf.Vec3f(0.0, 0.0, float(p.pose.get("yaw_deg", 0.0))))

        prim = mesh.GetPrim()
        UsdPhysics.CollisionAPI.Apply(prim)
        UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr(UsdPhysics.Tokens.none)

        visual, physics = materials[p.material]
        binding = UsdShade.MaterialBindingAPI.Apply(prim)
        binding.Bind(visual)
        binding.Bind(physics, UsdShade.Tokens.weakerThanDescendants, "physics")

        prim.CreateAttribute("terrain:type", Sdf.ValueTypeNames.String).Set(p.type)
        prim.CreateAttribute("terrain:maxSlopeDeg", Sdf.ValueTypeNames.Float).Set(
            p.summary["max_slope_deg"]
        )
        prim.CreateAttribute("terrain:p95RoughnessM", Sdf.ValueTypeNames.Float).Set(
            p.summary["p95_roughness_m"]
        )
        written.append(path)
    return written
