"""Remove only native owners directly intersecting the intended edit mask."""
import numpy as np


def remove_edited_owners(layers, intended):
    replacement = np.array(intended, dtype=bool, copy=True)
    counts = []
    for layer in layers:
        active = (layer['path'] != 0) | (layer['flags'] != 16)
        own_cells = np.arange(layer.size).reshape(layer.shape)
        owners = np.where(layer['path'] < 0, -layer['path'] - 1, own_cells)
        # Always select from the immutable requested footprint. Snapping an
        # irregular native owner to a new grid can otherwise select neighbours
        # repeatedly, erasing whole unedited connected areas.
        parents = np.unique(owners[intended & active])
        remove = np.isin(owners, parents) & active
        replacement |= remove
        counts.append({'owners_removed': int(len(parents)),
                       'owner_cells_removed': int(remove.sum())})
        layer[remove] = np.zeros((), dtype=layer.dtype)
        layer['flags'][remove] = 16
    return replacement, counts
