""" 
Set a naming convention for the Sentinel-1, ERS and Envisat scenes
Current naming convention:
Sentinel-1 masks: benchmark_data_CB/Sentinel-1/masks/S1A_EW_GRDM_1SDH_20170609T151156_20170609T151300_016958_01C3A7_84C0_Orb_NR_Cal_TC.tif
Sentinel-1 scenes: benchmark_data_CB/Sentinel-1/scenes/S1A_EW_GRDM_1SDH_20170609T151156_20170609T151300_016958_01C3A7_84C0_Orb_NR_Cal_TC.tif

ERS masks: benchmark_data_CB/ERS/masks/CG_prepro_SAR_IMP_1PNESA19961029_142649_00000018A016_00110_07980_0000_cropped.tif
    benchmark_data_CB/ERS/masks/PIG_prepro_SAR_IMP_1PNESA19950122_141125_00000018F142_01617_18418_0000_cropped.tif
    benchmark_data_CB/ERS/masks/CIS_prepro_SAR_IMP_1PNESA19960214_143603_00000016A008_00425_04287_0000_cropped.tif
    benchmark_data_CB/ERS/masks/PIG02_prepro_SAR_IMP_1PNESA20020105_135932_00000017A070_00167_35091_0000_cropped.tif
    benchmark_data_CB/ERS/masks/prepro_SAR_IMP_1PNESA19911203_091555_00000017A045_00007_01996_0000_cropped.tif
    benchmark_data_CB/ERS/masks/TG_prepro_SAR_IMP_1PNESA19951107_141549_00000018G151_00024_22557_0000_cropped.tif
ERS scenes: same filenames as masks but stored in scenes folder
ERS vectors: only has the associated date with the scene/mask as .shp, .prj, .dbf, .shx .qmd, .cpg

Envisat masks: benchmark_data_CB/Envisat/masks/prepro_ASA_IMP_1PNESA20060205_133956_000000182044_00482_20576_0000_cropped.tif
Envisat scenes: same
Envisat vectors: only has the associated date with the scene/mask as .shp, .prj, .dbf, .shx .qmd, .cpg

Names will follow:
[SATELLITE-ID]_[YYYYMMDD]_[polarization]_[extra]

"""

import os, re, csv
from pathlib import Path
from collections import defaultdict
from pathlib import Path

VECTOR_EXTS = {".shp",".shx",".dbf",".prj",".cpg",".qpj",".sbn",".sbx",".qix",".fix",".aih",".ain",".xml",".qmd"}
RASTER_EXTS = {".tif", ".tiff", ".img", ".vrt"}
ALLOWED_EXTS = VECTOR_EXTS | RASTER_EXTS

def is_hidden(p: Path) -> bool:
    return p.name.startswith(".") or p.name.startswith("~$")

# --- helpers --------------------------------------------------------------

def detect_family(path: Path) -> str | None:
    s = str(path)
    if any(k in s for k in ("Sentinel-1", "S1A_", "S1B_", "test_s1")):
        return "S1"
    if any(k in s for k in ("ERS", "test_ERS")):
        return "ERS"
    if any(k in s for k in ("Envisat", "ENVISAT", "test_envisat")):
        return "ENV"
    return None

def parse_date_any(name: str) -> str:
    # prefer 1PNESA<YYYYMMDD>, else any bare YYYYMMDD
    m = re.search(r'1PNESA(\d{8})', name)
    if m: return m.group(1)
    m = re.search(r'(\d{8})', name)
    return m.group(1) if m else "unknown_date"

def _last_code_token_any_ext(name: str) -> str:
    """
    Pick the last 6- or 4-char alphanumeric token between underscores,
    ending before '_' or '.' or end-of-string.
    Ignore common non-codes like GRDM/EW/1SDH/1SSH/MASK/CROP.
    Works for .tif, .tif.aux.xml, and shapefile sidecars.
    """
    IGNORE = {"GRDM", "EW", "1SDH", "1SSH", "MASK", "CROP"}
    pat = r'(?i)(?<=_)([A-Z0-9]{6}|[A-Z0-9]{4})(?=_|\.|$)'
    toks = [m.group(1).upper() for m in re.finditer(pat, name)]
    for t in reversed(toks):
        if t not in IGNORE:
            return t
    return "XXXX"

def s1_new_stem(filename: str) -> str:
    name = Path(filename).name
    # S1A/S1B
    mtype = re.search(r'(S1[AB])_EW_GRDM_1S[DS]H', name)
    s1_type = mtype.group(1) if mtype else "S1A"
    # Date
    mdate = re.search(r'_(\d{8})T\d{6}_', name)
    ymd = mdate.group(1) if mdate else parse_date_any(name)
    # Polarisation
    pol = "HH" if ("1SDH" in name or "1SSH" in name) else "UNK"
    # Code
    code = _last_code_token_any_ext(name)
    return f"{s1_type}_{ymd}_{pol}_EW_GRDM_1SDH_{code}"


def _id6_after_1PNESA(filename: str) -> str:
    """
    Get the 6-digit identifier immediately after 1PNESA<YYYYMMDD>_
    e.g. 1PNESA20060205_133956_...  -> '133956'
    """
    m = re.search(r'1PNESA\d{8}_(\d{6})', filename)
    if m:
        return m.group(1)
    # fallback: any 6-digit run
    m2 = re.search(r'(\d{6})', filename)
    return m2.group(1) if m2 else "XXXXXX"

def ers_new_stem(filename: str) -> str:
    ymd = parse_date_any(filename)
    pol = "VV"
    id6 = _id6_after_1PNESA(filename)
    return f"ERS_{ymd}_{pol}_{id6}"

def env_new_stem(filename: str) -> str:
    ymd = parse_date_any(filename)
    pol = "VV"
    id6 = _id6_after_1PNESA(filename)
    return f"ENV_{ymd}_{pol}_{id6}"

def new_stem_for(path: Path, family: str) -> str:
    name = path.name
    if family == "S1":   return s1_new_stem(name)
    if family == "ERS":  return ers_new_stem(name)
    if family == "ENV":  return env_new_stem(name)
    return f"UNK_{parse_date_any(name)}_UNK_X"

def is_vector_part(p: Path) -> bool:
    return p.suffix.lower() in VECTOR_EXTS

def group_vectors(root: Path):
    """Return dict[(dir, stem)] -> [files...] for shapefile datasets."""
    groups = defaultdict(list)
    for p in root.rglob("*"):
        if p.is_file() and is_vector_part(p):
            groups[(p.parent, p.stem)].append(p) 
        if not p.is_file():
            continue
        if is_hidden(p):
            continue  # skip .DS_Store 

        # only handle raster-ish files here
        if p.suffix.lower() not in RASTER_EXTS:
            continue
    # sort parts so .shp goes last (nicer UX during rename)
    for k, parts in groups.items():
        parts.sort(key=lambda q: (q.suffix.lower() == ".shp"))
    return groups

# --- main ----------------------------------------------------------------

def rename_everything(base_dir: str,
                      dry_run: bool = True,
                      resolve_conflicts: str = "suffix",  # "suffix" | "abort" | "skip"
                      plan_csv: str = "rename_plan.csv",
                      log_csv: str = "renamed_log_all.csv"):
    base = Path(base_dir)

    # 1) Build a plan (old_path -> new_path), grouping shapefiles
    plan = []
    taken = defaultdict(list)

    vector_groups = group_vectors(base)
    seen_vector_keys = set(vector_groups.keys())

    # queue vector datasets
    for (parent, stem), parts in vector_groups.items():
        fam = detect_family(parent)
        if not fam: continue
        stem_new = new_stem_for(parts[0], fam)
        for part in parts:
            dst = parent / f"{stem_new}{part.suffix}"
            plan.append((part, dst))
            taken[str(dst)].append(str(part))

    # queue all other files (non-vector)
    for p in base.rglob("*"):
        if not p.is_file(): 
            continue
        key = (p.parent, p.stem)
        if key in seen_vector_keys:
            # already handled as a dataset
            continue
        fam = detect_family(p)
        if not fam:
            continue
        stem_new = new_stem_for(p, fam)
        dst = p.with_name(f"{stem_new}{p.suffix}")
        plan.append((p, dst))
        taken[str(dst)].append(str(p))

    # 2) Detect conflicts / existing destinations
    conflicts = {dst: srcs for dst, srcs in taken.items() if len(srcs) > 1}
    existing = [(src, dst) for src, dst in plan if dst.exists()]

    # 3) Save plan CSV
    with open(plan_csv, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["from", "to"]); w.writerows([(str(a), str(b)) for a,b in plan])

    if conflicts or existing:
        if resolve_conflicts == "abort":
            print("ABORTING: conflicts or existing targets detected.")
            if conflicts:
                print("\nMultiple files map to the same destination:")
                for d, srcs in conflicts.items():
                    print("  ", d)
                    for s in srcs: print("    -", s)
            if existing:
                print("\nTargets already exist on disk:")
                for s,d in existing: print("   ", d, "<-", s)
            print(f"\nReview the plan: {plan_csv}")
            return
        elif resolve_conflicts in ("suffix","skip"):
            fixed = []
            counts = defaultdict(int)
            seen = set()
            for src, dst in plan:
                final = dst
                if (str(dst) in conflicts) or dst.exists():
                    if resolve_conflicts == "skip":
                        print(f"SKIP (conflict/existing): {src} -> {dst}")
                        continue
                    stem, suf = dst.stem, dst.suffix
                    while final.exists() or str(final) in seen:
                        counts[str(dst)] += 1
                        final = dst.with_name(f"{stem}_dup{counts[str(dst)]}{suf}")
                fixed.append((src, final))
                seen.add(str(final))
            plan = fixed

    # 4) Execute (or dry-run)
    if dry_run:
        for s,d in plan:
            print(f"Would rename: {s} -> {d}")
        print(f"\nDry run only. Plan saved: {plan_csv}")
        return

    for s,d in plan:
        if d.exists():
            raise FileExistsError(f"Refusing to overwrite existing file: {d}")
        os.rename(s, d)
        print(f"Renamed: {s} -> {d}")

    with open(log_csv, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["from","to"]); w.writerows([(str(a), str(b)) for a,b in plan])
    print(f"Done. Log saved: {log_csv}")

# --- example usage -------------------------------------------------------
if __name__ == "__main__":
    base = "/gws/nopw/j04/iecdt/amorgan/benchmark_data_CB/ICE-BENCH"
    # First: preview
    rename_everything(base, dry_run=True, resolve_conflicts="suffix", plan_csv="dry_run_plan.csv")
    # Then: actually do it
    # rename_everything(base, dry_run=False, resolve_conflicts="suffix")
