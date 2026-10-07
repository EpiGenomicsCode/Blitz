"""Standalone reconstruction of the SynthID Bio structure detector."""

from __future__ import annotations

import struct
from collections.abc import Iterable
from itertools import groupby
from pathlib import Path

import numpy as np

DETECTOR_PREFIX = "diffuser/~/watermark_detector/point_net/"
MAX_TOKENS = 2048

_LAYER_NAMES = tuple(
    "watermark_point_net_conv" + (f"_{index}" if index else "") for index in range(5)
)
_OUTPUT_NAME = "watermark_detector_logits"

_PROTEIN_ATOMS = {
    "ALA": ("N", "CA", "C", "O", "CB", "OXT"),
    "ARG": ("N", "CA", "C", "O", "CB", "CG", "CD", "NE", "CZ", "NH1", "NH2", "OXT"),
    "ASN": ("N", "CA", "C", "O", "CB", "CG", "OD1", "ND2", "OXT"),
    "ASP": ("N", "CA", "C", "O", "CB", "CG", "OD1", "OD2", "OXT"),
    "CYS": ("N", "CA", "C", "O", "CB", "SG", "OXT"),
    "GLN": ("N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "NE2", "OXT"),
    "GLU": ("N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "OE2", "OXT"),
    "GLY": ("N", "CA", "C", "O", "OXT"),
    "HIS": ("N", "CA", "C", "O", "CB", "CG", "ND1", "CD2", "CE1", "NE2", "OXT"),
    "ILE": ("N", "CA", "C", "O", "CB", "CG1", "CG2", "CD1", "OXT"),
    "LEU": ("N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "OXT"),
    "LYS": ("N", "CA", "C", "O", "CB", "CG", "CD", "CE", "NZ", "OXT"),
    "MET": ("N", "CA", "C", "O", "CB", "CG", "SD", "CE", "OXT"),
    "MSE": ("N", "CA", "C", "O", "CB", "CG", "SE", "CE", "OXT"),
    "PHE": ("N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OXT"),
    "PRO": ("N", "CA", "C", "O", "CB", "CG", "CD", "OXT"),
    "SER": ("N", "CA", "C", "O", "CB", "OG", "OXT"),
    "THR": ("N", "CA", "C", "O", "CB", "OG1", "CG2", "OXT"),
    "TRP": (
        "N",
        "CA",
        "C",
        "O",
        "CB",
        "CG",
        "CD1",
        "CD2",
        "NE1",
        "CE2",
        "CE3",
        "CZ2",
        "CZ3",
        "CH2",
        "OXT",
    ),
    "TYR": (
        "N",
        "CA",
        "C",
        "O",
        "CB",
        "CG",
        "CD1",
        "CD2",
        "CE1",
        "CE2",
        "CZ",
        "OH",
        "OXT",
    ),
    "UNK": ("N", "CA", "C", "O", "OXT"),
    "VAL": ("N", "CA", "C", "O", "CB", "CG1", "CG2", "OXT"),
}

_RNA_BACKBONE = (
    "OP3",
    "P",
    "OP1",
    "OP2",
    "O5'",
    "C5'",
    "C4'",
    "O4'",
    "C3'",
    "O3'",
    "C2'",
    "O2'",
    "C1'",
)
_DNA_BACKBONE = (
    "OP3",
    "P",
    "OP1",
    "OP2",
    "O5'",
    "C5'",
    "C4'",
    "O4'",
    "C3'",
    "O3'",
    "C2'",
    "C1'",
)
_NUCLEIC_ATOMS = {
    "A": _RNA_BACKBONE + ("N9", "C8", "N7", "C5", "C6", "N6", "N1", "C2", "N3", "C4"),
    "C": _RNA_BACKBONE + ("N1", "C2", "O2", "N3", "C4", "N4", "C5", "C6"),
    "G": _RNA_BACKBONE
    + ("N9", "C8", "N7", "C5", "C6", "O6", "N1", "C2", "N2", "N3", "C4"),
    "N": _RNA_BACKBONE,
    "U": _RNA_BACKBONE + ("N1", "C2", "O2", "N3", "C4", "O4", "C5", "C6"),
    "DA": _DNA_BACKBONE + ("N9", "C8", "N7", "C5", "C6", "N6", "N1", "C2", "N3", "C4"),
    "DC": _DNA_BACKBONE + ("N1", "C2", "O2", "N3", "C4", "N4", "C5", "C6"),
    "DG": _DNA_BACKBONE
    + ("N9", "C8", "N7", "C5", "C6", "O6", "N1", "C2", "N2", "N3", "C4"),
    "DN": _DNA_BACKBONE,
    "DT": _DNA_BACKBONE + ("N1", "C2", "O2", "N3", "C4", "O4", "C5", "C7", "C6"),
}

_PEPTIDE_TYPES = {"polypeptide(L)", "polypeptide(D)", "cyclic-pseudo-peptide"}
_NUCLEIC_TYPES = {
    "polyribonucleotide",
    "polydeoxyribonucleotide",
    "polydeoxyribonucleotide/polyribonucleotide hybrid",
    "other",
}


def _read_exact(stream, size: int, description: str) -> bytes:
    value = stream.read(size)
    if len(value) != size:
        raise ValueError(f"Incomplete {description} in checkpoint")
    return value


def load_detector_weights(checkpoint: str | Path):
    """Read only the watermark-detector records from a full AF3 checkpoint."""
    import jax
    import jax.numpy as jnp

    checkpoint = Path(checkpoint)
    if checkpoint.suffix == ".zst":
        raise ValueError("Decompress the checkpoint before scoring")

    layers: dict[str, dict[str, np.ndarray]] = {}
    header_size = struct.calcsize("<5i")
    with checkpoint.open("rb") as stream:
        while True:
            header = stream.read(header_size)
            if not header:
                break
            if len(header) != header_size:
                raise ValueError("Incomplete checkpoint record header")
            scope_len, name_len, dtype_len, ndim, data_len = struct.unpack(
                "<5i", header
            )
            if min(scope_len, name_len, dtype_len, ndim, data_len) < 0 or ndim > 32:
                raise ValueError("Invalid checkpoint record header")
            scope = _read_exact(stream, scope_len, "scope").decode("utf-8")
            name = _read_exact(stream, name_len, "parameter name").decode("utf-8")
            dtype_name = _read_exact(stream, dtype_len, "dtype").decode("utf-8")
            shape_bytes = _read_exact(stream, 4 * ndim, "shape")
            shape = struct.unpack(f"<{ndim}i", shape_bytes) if ndim else ()
            if any(size < 0 for size in shape):
                raise ValueError("Invalid checkpoint tensor shape")

            if not scope.startswith(DETECTOR_PREFIX):
                stream.seek(data_len, 1)
                continue

            dtype = np.dtype(dtype_name)
            if int(np.prod(shape, dtype=np.int64)) * dtype.itemsize != data_len:
                raise ValueError(f"Invalid byte count for {scope}/{name}")
            data = _read_exact(stream, data_len, f"data for {scope}/{name}")
            value = np.frombuffer(data, dtype=dtype).reshape(shape).copy()
            layer_name = scope[len(DETECTOR_PREFIX) :]
            layer = layers.setdefault(layer_name, {})
            if name in layer:
                raise ValueError(f"Duplicate detector parameter: {scope}/{name}")
            layer[name] = value

    expected_layers = set(_LAYER_NAMES) | {_OUTPUT_NAME}
    if set(layers) != expected_layers:
        raise ValueError(f"Unexpected detector scopes: {sorted(layers)}")
    for index, layer_name in enumerate(_LAYER_NAMES):
        layer = layers[layer_name]
        expected_shape = (11, 65 if index == 0 else 288, 288)
        if (
            set(layer) != {"w", "b"}
            or layer["w"].shape != expected_shape
            or layer["b"].shape != (288,)
        ):
            raise ValueError(f"Unexpected parameters for {layer_name}")
    output = layers[_OUTPUT_NAME]
    if set(output) != {"weights"} or output["weights"].shape != (288, 1):
        raise ValueError("Unexpected detector output layer")
    if not all(
        np.all(np.isfinite(value))
        for layer in layers.values()
        for value in layer.values()
    ):
        raise ValueError("Detector parameters contain non-finite values")
    return jax.tree.map(jnp.asarray, layers)


def geometry_features(positions, atom_mask, *, torsion_sign: float = 1.0):
    """Compute the geometric features used by our evaluation scorer."""
    import jax.numpy as jnp

    positions = jnp.asarray(positions, dtype=jnp.float32)
    atom_mask = jnp.asarray(atom_mask, dtype=jnp.float32)
    if positions.shape[-2:] != (24, 3) or atom_mask.shape != positions.shape[:-1]:
        raise ValueError("Expected positions [..., tokens, 24, 3] and a matching mask")
    positions = jnp.where(atom_mask[..., None] > 0, positions, 0.0)
    edges = positions[..., 1:, :] - positions[..., :-1, :]
    distance_mask = atom_mask[..., 1:] * atom_mask[..., :-1]
    distances = (
        jnp.sqrt(jnp.maximum(jnp.sum(edges * edges, axis=-1), 1e-8)) * distance_mask
    )
    a, b, c = edges[..., :-2, :], edges[..., 1:-1, :], edges[..., 2:, :]
    normal1, normal2 = jnp.cross(a, b), jnp.cross(b, c)
    cos_arg = jnp.sum(normal1 * normal2, axis=-1)
    sin_arg = jnp.sum(jnp.cross(normal1, normal2) * b, axis=-1)
    sin_arg /= jnp.sqrt(jnp.maximum(jnp.sum(b * b, axis=-1), 1e-8))
    angle = jnp.arctan2(torsion_sign * sin_arg, cos_arg)
    torsion_mask = (
        atom_mask[..., :-3]
        * atom_mask[..., 1:-2]
        * atom_mask[..., 2:-1]
        * atom_mask[..., 3:]
    )
    features = jnp.concatenate(
        [distances, jnp.sin(angle) * torsion_mask, jnp.cos(angle) * torsion_mask],
        axis=-1,
    )
    residue_mask = jnp.any(atom_mask > 0, axis=-1)
    return features, residue_mask


def detector_logits(weights, positions, atom_mask, *, torsion_sign: float = 1.0):
    """Return the uncalibrated detector logit for each input structure."""
    import jax
    import jax.numpy as jnp

    features, residue_mask = geometry_features(
        positions, atom_mask, torsion_sign=torsion_sign
    )
    if features.ndim == 2:
        features, residue_mask = features[None], residue_mask[None]
    if features.shape[1] > MAX_TOKENS:
        raise ValueError(f"Structures above {MAX_TOKENS} tokens are not supported")
    width = MAX_TOKENS - features.shape[1]
    x = jnp.pad(features, ((0, 0), (0, width), (0, 0)))
    mask = jnp.pad(residue_mask, ((0, 0), (0, width)))[..., None]
    for layer_name in _LAYER_NAMES:
        layer = weights[layer_name]
        x = jax.lax.conv_general_dilated(
            x,
            layer["w"],
            (1,),
            "SAME",
            dimension_numbers=("NWC", "WIO", "NWC"),
        )
        x = jax.nn.relu(x + layer["b"]) * mask
    pooled = jnp.sum(x, axis=1) / jnp.maximum(jnp.sum(mask, axis=1), 1)
    return (pooled @ weights[_OUTPUT_NAME]["weights"])[..., 0]


def _column(
    category: dict[str, list[str]], name: str, size: int | None = None
) -> list[str]:
    if name not in category:
        raise ValueError(f"mmCIF is missing _atom_site.{name}")
    values = category[name]
    if size is not None and len(values) != size:
        raise ValueError(f"Invalid _atom_site.{name} length")
    return values


def _is_missing(value: object) -> bool:
    return value is None or value is False or value in {"", ".", "?"}


def _dense_token(atoms: Iterable[tuple[str, np.ndarray]], order: tuple[str, ...]):
    positions = np.zeros((24, 3), dtype=np.float32)
    mask = np.zeros(24, dtype=np.float32)
    index_by_name = {name: index for index, name in enumerate(order)}
    if len(order) > 24:
        raise ValueError("Residue atom layout exceeds 24 slots")
    for name, xyz in atoms:
        if name not in index_by_name:
            raise ValueError(
                f"Atom {name!r} is not in the inferred standard-residue layout"
            )
        index = index_by_name[name]
        if mask[index]:
            raise ValueError(f"Duplicate atom {name!r} in residue")
        positions[index] = xyz
        mask[index] = 1
    return positions, mask


def coordinates_from_mmcif(cif_path: str | Path):
    """Reconstruct AF3's 24-atom token layout from a prediction mmCIF."""
    import gemmi

    cif_path = Path(cif_path)
    block = gemmi.cif.read_file(str(cif_path)).sole_block()
    atom_site = block.get_mmcif_category("_atom_site.")
    if not atom_site:
        raise ValueError(f"No _atom_site records in {cif_path}")
    atom_names = _column(atom_site, "label_atom_id")
    size = len(atom_names)
    elements = _column(atom_site, "type_symbol", size)
    residue_names = _column(atom_site, "label_comp_id", size)
    chain_ids = _column(atom_site, "label_asym_id", size)
    entity_ids = _column(atom_site, "label_entity_id", size)
    label_seq_ids = _column(atom_site, "label_seq_id", size)
    auth_seq_ids = _column(atom_site, "auth_seq_id", size)
    xs = _column(atom_site, "Cartn_x", size)
    ys = _column(atom_site, "Cartn_y", size)
    zs = _column(atom_site, "Cartn_z", size)
    alt_ids = atom_site.get("label_alt_id", ["."] * size)
    model_ids = atom_site.get("pdbx_PDB_model_num", ["1"] * size)
    if len(alt_ids) != size or len(model_ids) != size:
        raise ValueError("Invalid alternate-location or model columns")
    first_model = model_ids[0]

    entity_category = block.get_mmcif_category("_entity.")
    entity_types = dict(
        zip(
            entity_category.get("id", []),
            entity_category.get("type", []),
            strict=True,
        )
    )
    poly_category = block.get_mmcif_category("_entity_poly.")
    polymer_types = dict(
        zip(
            poly_category.get("entity_id", []),
            poly_category.get("type", []),
            strict=True,
        )
    )

    rows = []
    for index in range(size):
        if model_ids[index] != first_model:
            continue
        if not _is_missing(alt_ids[index]):
            raise ValueError("Alternate atom locations are not supported")
        if elements[index].upper() in {"H", "D"}:
            continue
        xyz = np.array([xs[index], ys[index], zs[index]], dtype=np.float32)
        if not np.all(np.isfinite(xyz)):
            raise ValueError(f"Non-finite coordinates in {cif_path}")
        entity_id = entity_ids[index]
        chain_type = polymer_types.get(entity_id, entity_types.get(entity_id, ""))
        residue_id = label_seq_ids[index]
        if _is_missing(residue_id):
            residue_id = auth_seq_ids[index]
        rows.append(
            (
                (chain_ids[index], residue_id, residue_names[index]),
                chain_type,
                residue_names[index],
                atom_names[index],
                xyz,
            )
        )

    token_positions = []
    token_masks = []
    for _, residue_rows in groupby(rows, key=lambda row: row[0]):
        residue_rows = list(residue_rows)
        chain_type = residue_rows[0][1]
        residue_name = residue_rows[0][2]
        atoms = [(row[3], row[4]) for row in residue_rows]
        if chain_type in _PEPTIDE_TYPES and residue_name in _PROTEIN_ATOMS:
            positions, mask = _dense_token(atoms, _PROTEIN_ATOMS[residue_name])
            token_positions.append(positions)
            token_masks.append(mask)
        elif chain_type in _NUCLEIC_TYPES and residue_name in _NUCLEIC_ATOMS:
            positions, mask = _dense_token(atoms, _NUCLEIC_ATOMS[residue_name])
            token_positions.append(positions)
            token_masks.append(mask)
        else:
            for _, xyz in atoms:
                positions = np.zeros((24, 3), dtype=np.float32)
                mask = np.zeros(24, dtype=np.float32)
                positions[0] = xyz
                mask[0] = 1
                token_positions.append(positions)
                token_masks.append(mask)

    if not token_positions:
        raise ValueError(f"No supported heavy atoms in {cif_path}")
    if len(token_positions) > MAX_TOKENS:
        raise ValueError(
            f"Structure has {len(token_positions)} tokens; maximum is {MAX_TOKENS}"
        )
    return np.stack(token_positions), np.stack(token_masks)
