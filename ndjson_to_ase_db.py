"""
Converts an NDJSON (one JSON per line) with Pymatgen structures
to an **ASE .db** (SQLite) database — 1 structure per row.

Requirements:
    pip install ase pymatgen tqdm
Usage:
    python ndjson_to_ase_db.py db.ndjson -o mp_2d.db --tag 2D --skip-errors

New features in this version:
- More robust: tries multiple structure keys (structure/final_structure/initial_structure)
  and has a manual fallback if Structure.from_dict fails.
- Stable typing for metadata: converts numeric strings ("222", "-1", "3.14") to int/float
  in known keys, avoiding ASE/SQLite warnings and data loss.
- Safely ignores inconsistent properties (e.g., None magmom, different sizes).
- Error report by category at the end for inspection.

"""
from __future__ import annotations
import argparse
import gzip
import io
import json
import math
import re
import sys
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Tuple

from ase.db import connect  # type: ignore
from pymatgen.core import Structure, Lattice  # type: ignore
from pymatgen.io.ase import AseAtomsAdaptor  # type: ignore

try:
    from tqdm import tqdm  # type: ignore
    TQDM = True
except Exception:
    TQDM = False

# Type mapping for known columns (forces consistent coercion)
FIELD_TYPES = {
    "nelements": int,
    "sg_number": int,
    "bandgap": float,
    "total_magnetization": float,
    "is_metal": bool,
    "energy": float,
    "energy_per_atom": float,
    "energy_vdw": float,
    "energy_vdw_per_atom": float,
    "exfoliation_energy_per_atom": float,
    "decomposition_energy": float,
}

# Spacegroup subfields to promote
SPACEGROUP_SUB = ("symbol", "number", "crystal_system", "point_group", "hall")

NUM_INT_RE = re.compile(r"^[+-]?[0-9]+$")
NUM_FLOAT_RE = re.compile(r"^[+-]?(?:[0-9]+\.?[0-9]*|\.[0-9]+)(?:[eE][+-]?[0-9]+)?$")


def parse_special_num_str(s: str):
    sl = s.strip().lower()
    if sl in ("inf", "+inf", "infinity", "+infinity"):
        return float("inf")
    if sl in ("-inf", "-infinity"):
        return -float("inf")
    if sl in ("nan", "+nan", "-nan"):
        return float("nan")
    return None


def open_maybe_gz(path: str):
    return io.TextIOWrapper(gzip.open(path, "rb"), encoding="utf-8") if path.endswith(".gz") else open(path, "r", encoding="utf-8")


def coerce_type(key: str, val: Any) -> Any:
    """Coerces `val` to a consistent type based on FIELD_TYPES.
    If the target is bool, handles 0/1/"true"/"false".
    If string looks like a number, converts it.
    Otherwise, returns as is.
    """
    if key in FIELD_TYPES:
        t = FIELD_TYPES[key]
        if t is bool:
            if isinstance(val, str):
                v = val.strip().lower()
                if v in {"true", "1", "yes"}:
                    return True
                if v in {"false", "0", "no"}:
                    return False
            return bool(val)
        if t is int:
            if isinstance(val, str) and NUM_INT_RE.match(val.strip()):
                return int(val)
            if isinstance(val, (float, int)) and not (isinstance(val, float) and (math.isnan(val) or math.isinf(val))):
                return int(val)
        if t is float:
            if isinstance(val, str):
                s = val.strip()
                if NUM_FLOAT_RE.match(s):
                    try:
                        return float(s)
                    except Exception:
                        pass
                sp = parse_special_num_str(s)
                if sp is not None:
                    return sp
            if isinstance(val, (int, float)):
                return float(val)
        # fallback
        return val
        if isinstance(val, (int, float)):
                return float(val)
        return val
    return val


def scalarize(x: Any) -> Any:
    """Converts values to simple types accepted by ASE DB.
    - numbers/strings/bools/None: return as is
    - lists/tuples/dicts: become compact JSON
    """
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    if isinstance(x, (list, tuple)):
        try:
            return json.dumps(x, ensure_ascii=False, separators=(",", ":"))
        except Exception:
            return str(x)
    if isinstance(x, dict):
        try:
            return json.dumps(x, ensure_ascii=False, separators=(",", ":"))
        except Exception:
            return str(x)
    return str(x)


def coerce_numeric_strings_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in d.items():
        if isinstance(v, str):
            s = v.strip()
            if NUM_INT_RE.match(s):
                try:
                    v = int(s)
                except Exception:
                    pass
            elif NUM_FLOAT_RE.match(s):
                try:
                    v = float(s)
                except Exception:
                    pass
            else:
                sp = parse_special_num_str(s)
                if sp is not None:
                    v = sp
        out[k] = v
    return out


def extract_kv(rec: Dict[str, Any]) -> Dict[str, Any]:
    kv: Dict[str, Any] = {}
    keys = [
        "material_id",
        "relative_id",
        "source_id",
        "creation_task_label",
        "discovery_process",
        "formula_pretty",
        "formula_reduced_abc",
        "formula_anonymous",
        "chemsys",
        "nelements",
        "sg_number",
        "sg_symbol",
        "bandgap",
        "total_magnetization",
        "is_metal",
        "energy",
        "energy_per_atom",
        "energy_vdw",
        "energy_vdw_per_atom",
        "exfoliation_energy_per_atom",
        "decomposition_energy",
    ]
    for k in keys:
        if k in rec:
            kv[k] = coerce_type(k, rec[k])

    # spacegroup
    sg = rec.get("spacegroup")
    if isinstance(sg, dict):
        for sub in SPACEGROUP_SUB:
            if sub in sg:
                key = f"sg_{sub}"
                kv[key] = coerce_type(key, sg[sub])

    # elements -> string
    if isinstance(rec.get("elements"), list):
        try:
            kv["elements"] = ",".join(map(str, rec["elements"]))
        except Exception:
            pass

    # created_at ISO
    created_at = rec.get("created_at")
    if isinstance(created_at, dict) and "$date" in created_at:
        kv["created_at"] = str(created_at["$date"])

    # preferred uid
    uid = (
        rec.get("material_id")
        or rec.get("relative_id")
        or (rec.get("_id", {}).get("$oid") if isinstance(rec.get("_id"), dict) else None)
    )
    if uid:
        kv["uid"] = str(uid)

    return {k: scalarize(v) for k, v in kv.items()}


def site_magmoms_from_struct_dict(struct_d: Dict[str, Any]) -> Optional[List[float]]:
    mags: List[float] = []
    for s in struct_d.get("sites", []) or []:
        props = s.get("properties") or {}
        m = props.get("magmom", 0.0)
        try:
            if m is None:
                m = 0.0
            mags.append(float(m))
        except Exception:
            mags.append(0.0)
    return mags if mags else None


def get_structure_dict(rec: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    for k in ("structure", "final_structure", "initial_structure"):
        v = rec.get(k)
        if isinstance(v, dict):
            return v
    return None


def build_structure_fallback(struct_d: Dict[str, Any]) -> Structure:
    # Lattice
    lat: Optional[Lattice] = None
    if isinstance(struct_d.get("lattice"), dict):
        L = struct_d["lattice"]
        if "matrix" in L and isinstance(L["matrix"], list):
            lat = Lattice(L["matrix"])
        else:
            a = float(L.get("a")); b = float(L.get("b")); c = float(L.get("c"))
            alpha = float(L.get("alpha", 90.0)); beta = float(L.get("beta", 90.0)); gamma = float(L.get("gamma", 90.0))
            lat = Lattice.from_parameters(a, b, c, alpha, beta, gamma)
    else:
        raise ValueError("structure without lattice")

    # Sites
    species: List[str] = []
    fracs: List[Tuple[float, float, float]] = []
    site_props: Dict[str, List[Any]] = {}

    for s in struct_d.get("sites", []) or []:
        el = None
        if isinstance(s.get("species"), list) and s["species"]:
            el = s["species"][0].get("element")
        if not el:
            el = s.get("label")
        if not el:
            raise ValueError("site without element")
        species.append(str(el))

        if "abc" in s and isinstance(s["abc"], list):
            x, y, z = s["abc"]
        elif "xyz" in s and isinstance(s["xyz"], list):
            # converts cartesian -> fractional
            x, y, z = lat.get_fractional_coords(s["xyz"])  # type: ignore
        else:
            raise ValueError("site without coordinates")
        fracs.append((float(x), float(y), float(z)))

        # site properties (e.g., magmom)
        props = s.get("properties") or {}
        for pk, pv in props.items():
            site_props.setdefault(pk, []).append(pv)

    return Structure(lat, species, fracs, site_properties=site_props if site_props else None)


def process(in_path: str, out_db: str, tag: Optional[str] = None, skip_errors: bool = False) -> None:
    adaptor = AseAtomsAdaptor()
    written = 0
    failed = 0
    total = 0
    err_counter: Counter[str] = Counter()

    with connect(out_db) as db, open_maybe_gz(in_path) as fh:
        iterator: Iterable[str] = fh
        if TQDM:
            try:
                iterator = tqdm(fh, desc="Reading NDJSON", unit="lin")
            except Exception:
                pass

        for line_no, line in enumerate(iterator, 1):
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                rec = json.loads(line)

                struct_d = get_structure_dict(rec)
                if not struct_d:
                    raise ValueError("record without structure field")

                # Tries the standard path
                try:
                    pmg_struct = Structure.from_dict(struct_d)
                except Exception:
                    # Robust fallback
                    pmg_struct = build_structure_fallback(struct_d)

                atoms = adaptor.get_atoms(pmg_struct)

                # magmoms
                mags = site_magmoms_from_struct_dict(struct_d)
                if mags and len(mags) == len(atoms):
                    try:
                        atoms.set_initial_magnetic_moments(mags)
                    except Exception:
                        pass

                # metadata
                kv = coerce_numeric_strings_dict(extract_kv(rec))
                if tag:
                    kv["tag"] = tag

                # writes
                db.write(atoms, **kv)
                written += 1
            except Exception as e:
                failed += 1
                emsg = str(e)
                # categorizes error
                if "structure" in emsg:
                    err_counter["no_structure"] += 1
                elif "site" in emsg and "coord" in emsg:
                    err_counter["site_no_coord"] += 1
                elif "lattice" in emsg:
                    err_counter["no_lattice"] += 1
                else:
                    err_counter["others"] += 1
                msg = f"[WARN] line {line_no}: {e}"
                if skip_errors:
                    print(msg, file=sys.stderr)
                    continue
                else:
                    raise

    print(f"Completed. Total: {total} | Written: {written} | Failed: {failed}")
    if failed:
        print("Error categories:", dict(err_counter))


def main():
    ap = argparse.ArgumentParser(description="Converts NDJSON (with Pymatgen structures) to ASE .db")
    ap.add_argument("input", help="path to .ndjson file (optional .gz)")
    ap.add_argument("-o", "--out", default="mp_2d.db", help="output ASE .db (default: mp_2d.db)")
    ap.add_argument("--tag", default=None, help="optional tag to write in each record")
    ap.add_argument("--skip-errors", action="store_true", help="ignores lines with errors and continues")
    args = ap.parse_args()
    process(args.input, args.out, tag=args.tag, skip_errors=args.skip_errors)


if __name__ == "__main__":
    main()