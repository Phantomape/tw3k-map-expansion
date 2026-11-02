"""Keep the vertical skirts of the coarse land meshes intact when re-seating heights.

Native land_mesh_*.rigid_model_v2 files close the cracks between coarse terrain
blocks with skirts: every edge vertex has a partner at the same (x, z) exactly
1.0 lower. A height pass that writes one surface height to every vertex at a
position lifts the partner onto the top and the skirt vanishes; the cracks then
show the water plane beneath the land as thin blue lines all over the map.

Height passes compute the new *surface* height per vertex and call
seat_on_surface(), which keeps each vertex's offset below the surface.
"""
import numpy as np


SKIRT_DEPTH = 1.0


def skirt_offsets(v, faces=None):
    """-SKIRT_DEPTH on skirt-bottom vertices, 0 elsewhere.

    A skirt bottom sits exactly SKIRT_DEPTH below the highest vertex at its
    (x, z). Other coincident vertices (retained native points next to
    regenerated ones) are surface points and get 0, so re-seating flattens them
    onto the surface. With faces given, vertices used by a non-vertical face
    are surface points too.
    """
    key = np.round(np.asarray(v[:, [0, 2]], np.float64), 4)
    _, inv = np.unique(key, axis=0, return_inverse=True)
    inv = inv.ravel()
    y = np.asarray(v[:, 1], np.float64)
    top = np.full(inv.max() + 1, -np.inf)
    np.maximum.at(top, inv, y)
    skirt = np.abs(y - top[inv] + SKIRT_DEPTH) < 1e-2
    if faces is not None:
        f = np.asarray(faces).reshape(-1, 3)
        t = np.asarray(v[:, :3], np.float64)[f]
        area = np.abs((t[:, 1, 0] - t[:, 0, 0]) * (t[:, 2, 2] - t[:, 0, 2]) - (t[:, 2, 0] - t[:, 0, 0]) * (t[:, 1, 2] - t[:, 0, 2]))
        surface = np.zeros(len(y), bool)
        surface[f[area > 1e-6].ravel()] = True
        # With faces the topology decides: a vertex only on vertical faces and
        # well below a coincident point is a skirt bottom. Regenerated lattice
        # points can share its (x, z) at a slightly different surface height
        # (tile-aligned frame), which breaks the exact 1.0 drop test above.
        used = np.zeros(len(y), bool)
        used[f.ravel()] = True
        skirt = (skirt | (used & (top[inv] - y > SKIRT_DEPTH / 2))) & ~surface
    return np.where(skirt, -SKIRT_DEPTH, 0.).astype(np.float32)


def seat_on_surface(v, surface, mask, offsets=None, faces=None):
    """Set masked vertices to surface height plus their skirt offset (in place)."""
    if offsets is None:
        offsets = skirt_offsets(v, faces)
    v[mask, 1] = surface[mask] + offsets[mask]
    return v



# Open sea outside the native map: coarse triangles wholly at or below OCEAN_DROP_Y are left out.
# The canvas sea floor (SEA_FLOOR -2.445 in the LF) under rectangular coarse grids met the native
# sea meshes (-2.029) and the ocean background quad on straight lines (live 2048k). A triangle
# wholly at or below the background quad (extend_campaign_ocean_background.OCEAN_Y, the native
# floor) lies under it, so leaving it out changes nothing visible except that the open sea is
# one floor level, and its vertices are freed. Triangles reaching above it (shore slopes) stay.
OCEAN_DROP_Y = -2.029016


def open_sea_faces(v, faces):
    """Boolean per face (rows of `faces`, indices into `v` rows of raw x, y, z): every vertex at
    or below OCEAN_DROP_Y and the face centre outside the native map (raw tile coordinates)."""
    import campaign_world as _frame
    faces = np.asarray(faces).reshape(-1, 3)
    deep = (v[faces, 1] <= OCEAN_DROP_Y).all(axis=1)
    cx = _frame.tile_coord(v[faces, 0].mean(axis=1)); cz = _frame.tile_coord(v[faces, 2].mean(axis=1))
    outside = (cx < _frame.NATIVE_TILE_X0) | (cx >= _frame.NATIVE_TILE_X1) | (cz < _frame.NATIVE_TILE_Y0) | (cz >= _frame.NATIVE_TILE_Y1)
    return deep & outside
