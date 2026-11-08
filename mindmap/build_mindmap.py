"""Generate ``mindmap/graph_data.js`` for the interactive code map (``mindmap/index.html``).

Nodes   = every source file (Python modules, scripts, tests, YAML configs; package
          ``__init__`` files are folded into the modules they re-export).
Clusters = the architectural blocks defined in ``CLUSTERS`` below.
Edges   = real dependencies parsed from the code with ``ast``: imports (resolved
          through package re-exports) and config files read by name / via ``base:``.

File descriptions and function lists are taken from docstrings, so the map stays
accurate as the code changes. Re-run after adding or moving files:

    python mindmap/build_mindmap.py
"""
from __future__ import annotations

import ast
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "src" / "skinlesion"
OUT = Path(__file__).resolve().parent / "graph_data.js"

# ---------------------------------------------------------------------------
# Cluster definitions (the abstraction layer of the map)
# ---------------------------------------------------------------------------
CLUSTERS = {
    "scripts": {
        "name": "CLI entry points",
        "match": ["scripts/"],
        "description": "Thin command-line wrappers, one per pipeline stage. They hold no logic of their own: they "
                       "parse --config / --set, seed everything, and call into the library.",
        "functionalities": [
            "download_data: fetch HAM10000 (Dataverse or Kaggle)",
            "prepare_data: Step 1 class folders + 100-per-class subset",
            "run_ml: classical ML pipeline (Table 4, ML part of Table 5)",
            "run_cnn: customised CNN (Tables 6, 7, Fig. 5)",
            "run_all: end-to-end reproduction in one command",
            "compare_with_paper: rebuild the paper-vs-ours report",
            "Shared --config / --set KEY=VALUE overrides on every script",
        ],
    },
    "configs": {
        "name": "Experiment configs",
        "match": ["configs/"],
        "description": "YAML files that hold every hyper-parameter of the paper. A variation is a new YAML that "
                       "inherits from paper.yaml with `base:` and changes only a few keys.",
        "functionalities": [
            "paper.yaml: paper-faithful settings, with the table each number comes from",
            "Dataset paths, class list, sampling size, augmentation switches",
            "ML: image size, histogram bins, scaler, split, k-fold, Table 1 models",
            "CNN: Table 2 architecture blocks, Table 3 training hyper-parameters",
            "variations/no_leakage.yaml: example variation (flip twins kept on one side of the split)",
        ],
    },
    "utils": {
        "name": "Utilities",
        "match": ["src/skinlesion/utils/"],
        "description": "Shared plumbing used by every other block: configuration objects, reproducibility and "
                       "filesystem helpers.",
        "functionalities": [
            "Config: dict with dot access, nested overrides, YAML inheritance (base:)",
            "load_config: parse YAML and --set key.sub=value overrides",
            "set_seed: seed random / numpy / torch for reproducible runs",
            "Project-relative path resolution, ensure_dir",
            "JSON save/load with numpy type support",
        ],
    },
    "data": {
        "name": "Data layer",
        "match": ["src/skinlesion/data/"],
        "description": "Turns raw HAM10000 into the balanced, augmented image arrays both pipelines consume. "
                       "Implements the paper's Steps 1–3 and the train/test split.",
        "functionalities": [
            "Download HAM10000 from Harvard Dataverse or Kaggle and extract the zips",
            "Parse metadata (lesion_id, image_id, dx, …) and find images in any folder layout",
            "Step 1: organise all 10,015 images into one folder per diagnosis",
            "Sample 100 images per class (700) with a fixed seed + manifest.csv",
            "Step 3: offline horizontal-flip augmentation → 200 per class (1,400)",
            "Keras ImageDataGenerator-style random affine (rotate, shift, shear, zoom, flip)",
            "Step 2: load + resize to 220×220 (ML) or 96×96 (CNN)",
            "80/20 split: paper-faithful, or grouped so flip twins never leak",
        ],
    },
    "features": {
        "name": "Feature extraction",
        "match": ["src/skinlesion/features/"],
        "description": "Global feature descriptors for the classical ML models: one vector per image that describes "
                       "colour, texture and shape.",
        "functionalities": [
            "Colour: 8×8×8 HSV colour histogram (512 values)",
            "Texture: 13 Haralick features averaged over 4 directions (mahotas)",
            "Shape: 7 Hu moments (OpenCV)",
            "Concatenate into one 532-d global feature vector per image",
            "Cache features + labels in HDF5, as the paper does",
            "Pluggable DESCRIPTORS registry for new features",
        ],
    },
    "models": {
        "name": "Models",
        "match": ["src/skinlesion/models/"],
        "description": "Model factories built from the config: the seven classical classifiers of Table 1 and the "
                       "customised CNN of Table 2.",
        "functionalities": [
            "ML registry: LR, LDA, KNN, DT, RF, NB, SVM (+ GradientBoosting for variations)",
            "build_ml_models: instantiate each classifier with its Table 1 hyper-parameters",
            "SkinLesionCNN: Conv32 → 2×Conv64 → 2×Conv128 → Dense1024 → softmax(7)",
            "ConvBlock: Conv-ReLU-BN stacks + MaxPool + Dropout, defined in YAML",
            "Keras-equivalent initialisation (Glorot) and BatchNorm settings",
            "count_parameters (8,677,447 trainable)",
        ],
    },
    "training": {
        "name": "Training pipelines",
        "match": ["src/skinlesion/training/"],
        "description": "The two experiment loops that connect data, features, models and evaluation. This is the hub "
                       "of the codebase.",
        "functionalities": [
            "ML: extract or load cached features → MinMax scale → 80/20 split",
            "ML: 10-fold cross-validation accuracy per classifier (Table 4)",
            "ML: fit on train, weighted P/R/F1 on test (Table 5)",
            "CNN: dataloaders with on-the-fly augmentation",
            "CNN: Adam(1e-3) with Keras-style decay, batch 32, 150 epochs",
            "CNN: per-epoch train/test history, final model + metrics saved (Tables 6, 7)",
            "Writes outputs/ml and outputs/cnn (results.json, figures, model.pt)",
        ],
    },
    "evaluation": {
        "name": "Evaluation & reporting",
        "match": ["src/skinlesion/evaluation/"],
        "description": "Turns predictions into the paper's metrics and figures, and compares every number against "
                       "the values published in the paper.",
        "functionalities": [
            "Accuracy, weighted and macro precision/recall/F1, per-class report, confusion matrix",
            "Fig. 5-style training history, confusion-matrix and paper-vs-ours bar charts",
            "paper_reference: Tables 4–7 copied from the paper",
            "build_report: outputs/report.md with paper and our numbers side by side",
        ],
    },
    "tests": {
        "name": "Tests",
        "match": ["tests/"],
        "description": "Pytest suite that checks each block in isolation and the whole pipeline on a synthetic "
                       "mini-HAM10000, so swapping a block cannot silently break the others.",
        "functionalities": [
            "Synthetic HAM10000 fixture (84 generated images + metadata)",
            "Unit tests: config overrides, feature shape, flip, grouped split, augmenter, CNN size, model registry",
            "End-to-end smoke test: prepare → ML → CNN → report",
        ],
    },
}

CONFIG_FILES = sorted((ROOT / "configs").rglob("*.yaml"))


def rel(p: Path) -> str:
    return p.relative_to(ROOT).as_posix()


def cluster_of(relpath: str) -> str:
    for cid, c in CLUSTERS.items():
        if any(relpath.startswith(m) for m in c["match"]):
            return cid
    raise KeyError(relpath)


def clean(text: str) -> str:
    """Strip reStructuredText markup (``code``) so docstrings read as plain text."""
    return text.replace("``", "")


def first_paragraph(doc: str | None) -> str:
    if not doc:
        return ""
    para = doc.strip().split("\n\n")[0]
    return clean(" ".join(line.strip() for line in para.splitlines()))


def first_line(doc: str | None) -> str:
    return clean((doc or "").strip().splitlines()[0].strip()) if doc and doc.strip() else ""


def module_name(path: Path) -> str | None:
    """'src/skinlesion/data/dataset.py' -> 'skinlesion.data.dataset'."""
    try:
        parts = path.relative_to(ROOT / "src").with_suffix("").parts
    except ValueError:
        return None
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def collect_python_files():
    files = sorted(PKG.rglob("*.py")) + sorted((ROOT / "scripts").glob("*.py")) + sorted((ROOT / "tests").glob("*.py"))
    return [f for f in files if "__pycache__" not in f.parts]


def build():
    py_files = collect_python_files()
    mod_to_file = {module_name(f): f for f in py_files if module_name(f)}
    local_top = {f.stem: f for f in py_files if f.parent.name in ("scripts", "tests")}

    # re-exports in package __init__ files: package -> {symbol: module}
    reexports: dict[str, dict[str, str]] = {}
    for f in py_files:
        if f.name != "__init__.py":
            continue
        pkg = module_name(f)
        table = {}
        for node in ast.walk(ast.parse(f.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ImportFrom) and node.level == 1 and node.module:
                for a in node.names:
                    table[a.asname or a.name] = f"{pkg}.{node.module}"
        reexports[pkg] = table

    def resolve_from(f: Path, node: ast.ImportFrom) -> list[Path]:
        if node.level:
            base = module_name(f)
            pkg_parts = base.split(".") if f.name == "__init__.py" else base.split(".")[:-1]
            pkg_parts = pkg_parts[: len(pkg_parts) - (node.level - 1)]
            target = ".".join(pkg_parts + ([node.module] if node.module else []))
        else:
            target = node.module or ""
            if target in local_top:
                return [local_top[target]]
        hits = []
        for a in node.names:
            sub = f"{target}.{a.name}"
            if sub in mod_to_file:                       # `from pkg import submodule`
                hits.append(mod_to_file[sub])
            elif a.name in reexports.get(target, {}):     # re-exported symbol
                hits.append(mod_to_file[reexports[target][a.name]])
            elif target in mod_to_file:
                hits.append(mod_to_file[target])
        return hits

    nodes, edges = {}, set()

    for f in py_files:
        if f.name == "__init__.py":
            continue
        rp = rel(f)
        src = f.read_text(encoding="utf-8")
        tree = ast.parse(src)
        symbols = []
        for n in tree.body:
            if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and not n.name.startswith("_"):
                kind = "class" if isinstance(n, ast.ClassDef) else "def"
                if isinstance(n, ast.FunctionDef) and n.decorator_list and any(
                        getattr(d, "attr", getattr(d, "id", "")) == "fixture" for d in n.decorator_list):
                    kind = "fixture"
                symbols.append({"name": n.name, "kind": kind, "doc": first_line(ast.get_docstring(n))})
        consts = [t.id for n in tree.body if isinstance(n, ast.Assign) for t in n.targets
                  if isinstance(t, ast.Name) and t.id.isupper()]
        nodes[rp] = {"id": rp, "label": f.name, "cluster": cluster_of(rp), "kind": "python",
                     "doc": first_paragraph(ast.get_docstring(tree)), "symbols": symbols, "constants": consts,
                     "loc": sum(1 for line in src.splitlines() if line.strip())}
        for n in ast.walk(tree):
            if isinstance(n, ast.ImportFrom):
                for target in resolve_from(f, n):
                    t = rel(target)
                    if target.name == "__init__.py" or t == rp:
                        continue
                    edges.add((rp, t, "import"))
        for cfg in CONFIG_FILES:  # configs referenced by file name
            if re.search(rf'["/]{re.escape(cfg.name)}"', src):
                if cfg.parent.name == "variations" and "variations" not in src:
                    continue
                edges.add((rp, rel(cfg), "reads config"))

    for cfg in CONFIG_FILES:
        rp = rel(cfg)
        text = cfg.read_text(encoding="utf-8")
        comment = [line.lstrip("# ").rstrip() for line in text.splitlines() if line.startswith("#")]
        comment = " ".join(c for c in comment if c and not set(c) <= {"-"})
        sections = [m.group(1) for m in re.finditer(r"^([a-z_]+):", text, re.M) if m.group(1) != "base"]
        nodes[rp] = {"id": rp, "label": cfg.name, "cluster": "configs", "kind": "yaml", "doc": comment,
                     "symbols": [{"name": s, "kind": "section", "doc": ""} for s in sections], "constants": [],
                     "loc": sum(1 for line in text.splitlines() if line.strip() and not line.startswith("#"))}
        m = re.search(r"^base:\s*(\S+)", text, re.M)
        if m:
            edges.add((rp, rel((cfg.parent / m.group(1)).resolve()), "inherits"))

    edges = [{"source": s, "target": t, "kind": k} for s, t, k in sorted(edges) if s in nodes and t in nodes]
    clusters = []
    for cid, c in CLUSTERS.items():
        members = [n for n in nodes.values() if n["cluster"] == cid]
        clusters.append({"id": cid, "name": c["name"], "description": c["description"],
                         "functionalities": c["functionalities"], "files": [n["id"] for n in members],
                         "folder": c["match"][0], "loc": sum(n["loc"] for n in members)})
    return {"clusters": clusters, "nodes": list(nodes.values()), "edges": edges}


if __name__ == "__main__":
    graph = build()
    OUT.write_text("// generated by mindmap/build_mindmap.py — do not edit by hand\nwindow.GRAPH = "
                   + json.dumps(graph, indent=1, ensure_ascii=False) + ";\n", encoding="utf-8")
    print(f"{len(graph['clusters'])} clusters, {len(graph['nodes'])} files, {len(graph['edges'])} dependencies -> {OUT}")
