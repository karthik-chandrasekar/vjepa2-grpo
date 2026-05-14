"""Phase 1: download LIBERO data + cache HF model weights.

Single-pod sequential setup with H200 141GB. Run AFTER setup.sh + sanity.py.

Three independent groups (pass any combination):

    # everything (recommended for first run on a fresh pod)
    nohup python -u scripts/download_data.py --all > /workspace/logs/download.log 2>&1 &

    # just the LIBERO HDF5 demos
    python scripts/download_data.py --libero

    # just the HF model weights
    python scripts/download_data.py --models

    # just LIBERO-Plus (eval-only, can defer until D14)
    python scripts/download_data.py --libero-plus

Disk budget (LIBERO-only path):
    libero_spatial + libero_object + libero_goal + libero_10  ~  20 GB
    V-JEPA-2 ViT-g ~ 4 GB
    OpenVLA-OFT (libero-spatial)  ~ 14 GB
    OpenVLA-OFT (other 3 suites, optional)  ~ 42 GB
    LIBERO-Plus repo + assets  ~ 5 GB
    --- TOTAL ----  ~ 85 GB (LIBERO-only, single suite policy) / 130 GB (all 4 policies)

NB. Critical adjustment to flag against the source code we already wrote:
OpenVLA-OFT was finetuned with `num_images_in_input=2` (third-person +
wrist camera) and `use_proprio=True`. Our policy.py wrapper currently
passes only the agentview image. Before Phase 4 (GRPO), update
`policy.py::_predict_mean_action` to also pass the wrist image and proprio
through OFT's processor. The OpenVLA-OFT repo's `experiments/robot/openvla_utils.py`
has the canonical reference call.
"""
from __future__ import annotations
import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

WORKSPACE = Path(os.environ.get("WORKSPACE_ROOT", "/workspace"))
DATA_ROOT = WORKSPACE / "data"
LIBERO_ROOT = DATA_ROOT / "libero"
HF_CACHE = WORKSPACE / ".hf_cache"
LOG_ROOT = WORKSPACE / "logs"

# Default LIBERO source. `yifengzhu-hf/LIBERO-datasets` is the official mirror
# of the upstream University of Texas LIBERO HDF5s. `clip-rt/modified_libero_hdf5`
# is the OpenVLA-style modified version (success-filtered, re-rendered);
# strictly speaking that's closer to what OpenVLA-OFT was trained on, but the
# vanilla mirror is more widely used and well-tested.
LIBERO_HF_REPO = "yifengzhu-hf/LIBERO-datasets"
LIBERO_HF_REPO_OPENVLA = "clip-rt/modified_libero_hdf5"

LIBERO_SUITES_V1 = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
LIBERO_SUITES_FULL = LIBERO_SUITES_V1 + ["libero_90"]

VJEPA2_REPO = "facebook/vjepa2-vitg-fpc64-384"

OPENVLA_OFT_BASE = "moojink/openvla-7b-oft-finetuned-libero"
OPENVLA_OFT_SUITES = ["spatial", "object", "goal", "10"]

LIBERO_PLUS_REPO_URL = "https://github.com/sylvestf/LIBERO-plus.git"
LIBERO_PLUS_DIR = WORKSPACE / "LIBERO-plus"

LIBERO_REPO_URL = "https://github.com/Lifelong-Robot-Learning/LIBERO.git"
LIBERO_REPO_DIR = WORKSPACE / "LIBERO"


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def section(title: str):
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}", flush=True)


def free_gb(path: Path) -> float:
    s = shutil.disk_usage(path)
    return s.free / 1e9


def ensure(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def check_hf_login():
    """Verify the HF token is set (needed even for public repos to lift rate limits)."""
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not tok:
        print("[WARN] No HF_TOKEN set in env. Public repos will work but rate limits will be tight.")
        print("       export HF_TOKEN=<your_token>  (or run `huggingface-cli login`)")
        return False
    print(f"[hf] token detected (...{tok[-4:]})")
    return True


def shell(cmd: List[str], cwd: Optional[Path] = None, check: bool = True):
    print(f"$ {' '.join(cmd)}" + (f"   (cwd={cwd})" if cwd else ""), flush=True)
    return subprocess.run(cmd, cwd=cwd, check=check)


def t_human(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds/60:.1f}m"
    return f"{seconds/3600:.2f}h"


# ---------------------------------------------------------------------------
# LIBERO data
# ---------------------------------------------------------------------------

def download_libero(
    suites: List[str],
    use_openvla_modified: bool = False,
    revision: Optional[str] = None,
) -> None:
    section(f"LIBERO data ({'OpenVLA-modified' if use_openvla_modified else 'official mirror'})")

    repo = LIBERO_HF_REPO_OPENVLA if use_openvla_modified else LIBERO_HF_REPO
    ensure(LIBERO_ROOT)

    if free_gb(WORKSPACE) < 30:
        raise RuntimeError(
            f"Only {free_gb(WORKSPACE):.1f} GB free on {WORKSPACE}; need >= 30 GB for LIBERO."
        )

    from huggingface_hub import snapshot_download

    # Filter to only the suites we want, in HDF5 format.
    # Both repos use `<suite>/*.hdf5` layout.
    allow = []
    for s in suites:
        allow.append(f"{s}/*.hdf5")
        allow.append(f"{s}/*/*.hdf5")  # in case the OpenVLA-modified repo has subdirs

    print(f"[libero] repo: {repo}")
    print(f"[libero] suites: {suites}")
    print(f"[libero] dest: {LIBERO_ROOT}")
    print(f"[libero] free space: {free_gb(WORKSPACE):.1f} GB")

    t0 = time.time()
    path = snapshot_download(
        repo_id=repo,
        repo_type="dataset",
        local_dir=str(LIBERO_ROOT),
        allow_patterns=allow,
        revision=revision,
        max_workers=4,
        # use the network cache rather than the symlink-from-cache so the volume
        # is self-contained (no broken links if HF_HOME ever moves)
        local_dir_use_symlinks=False,
    )
    dt = time.time() - t0
    print(f"[libero] done in {t_human(dt)}; root: {path}")

    # Quick verify
    total_bytes = 0
    file_count = 0
    for suite in suites:
        sd = LIBERO_ROOT / suite
        if not sd.exists():
            print(f"[WARN] suite dir not found after download: {sd}")
            continue
        for f in sd.rglob("*.hdf5"):
            total_bytes += f.stat().st_size
            file_count += 1
    print(f"[libero] verified: {file_count} hdf5 files, {total_bytes/1e9:.1f} GB")


# ---------------------------------------------------------------------------
# HF model weights
# ---------------------------------------------------------------------------

def download_models(
    vjepa2: bool = True,
    openvla_oft_suites: Optional[List[str]] = None,
    openvla_combined: bool = False,
) -> None:
    section("HF model weights")
    if openvla_oft_suites is None:
        # Default: just the spatial checkpoint — the one referenced in configs/grpo.yaml.
        openvla_oft_suites = ["spatial"]

    from huggingface_hub import snapshot_download

    os.environ.setdefault("HF_HOME", str(HF_CACHE))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(HF_CACHE))
    ensure(HF_CACHE)

    targets = []
    if vjepa2:
        targets.append(("model", VJEPA2_REPO))
    for s in openvla_oft_suites:
        targets.append(("model", f"{OPENVLA_OFT_BASE}-{s}"))
    if openvla_combined:
        targets.append(("model", f"{OPENVLA_OFT_BASE}-spatial-object-goal-10"))

    print(f"[models] cache: {HF_CACHE}   free: {free_gb(WORKSPACE):.1f} GB")
    for repo_type, repo_id in targets:
        t0 = time.time()
        print(f"\n[models] -> {repo_id}")
        path = snapshot_download(
            repo_id=repo_id,
            repo_type=repo_type,
            cache_dir=str(HF_CACHE),
            max_workers=4,
        )
        # Sum size of the downloaded snapshot
        total = 0
        for f in Path(path).rglob("*"):
            if f.is_file():
                total += f.stat().st_size
        print(f"[models] {repo_id}: {total/1e9:.1f} GB  in {t_human(time.time()-t0)}")
        print(f"[models]   path: {path}")


# ---------------------------------------------------------------------------
# LIBERO Python package (sim env)
# ---------------------------------------------------------------------------

def install_libero_package(reinstall: bool = False) -> None:
    """Clone the LIBERO repo and pip install -e ."""
    section("LIBERO Python package")
    if LIBERO_REPO_DIR.exists() and not reinstall:
        print(f"[libero-pkg] {LIBERO_REPO_DIR} already exists; pulling latest")
        shell(["git", "pull"], cwd=LIBERO_REPO_DIR, check=False)
    else:
        if LIBERO_REPO_DIR.exists():
            shutil.rmtree(LIBERO_REPO_DIR)
        ensure(LIBERO_REPO_DIR.parent)
        shell(["git", "clone", "--depth", "1", LIBERO_REPO_URL, str(LIBERO_REPO_DIR)])
    shell([sys.executable, "-m", "pip", "install", "-e", "."], cwd=LIBERO_REPO_DIR)

    # Verify
    try:
        from libero.libero import benchmark
        bench = benchmark.get_benchmark_dict()["libero_spatial"]()
        print(f"[libero-pkg] OK; libero_spatial has {bench.n_tasks} tasks")
    except Exception as e:
        print(f"[libero-pkg] WARN: import failed: {e}")


# ---------------------------------------------------------------------------
# LIBERO-Plus
# ---------------------------------------------------------------------------

def install_libero_plus(reinstall: bool = False) -> None:
    section("LIBERO-Plus (perturbation eval)")
    if LIBERO_PLUS_DIR.exists() and not reinstall:
        print(f"[libero-plus] {LIBERO_PLUS_DIR} exists; pulling latest")
        shell(["git", "pull"], cwd=LIBERO_PLUS_DIR, check=False)
    else:
        if LIBERO_PLUS_DIR.exists():
            shutil.rmtree(LIBERO_PLUS_DIR)
        ensure(LIBERO_PLUS_DIR.parent)
        shell(["git", "clone", "--depth", "1", LIBERO_PLUS_REPO_URL, str(LIBERO_PLUS_DIR)])
    # pip install -e so vjepa2_grpo/eval_libero_plus.py can `import libero_plus`
    shell([sys.executable, "-m", "pip", "install", "-e", "."],
          cwd=LIBERO_PLUS_DIR, check=False)

    # Verify (some LIBERO-Plus dist data may be lazy-downloaded on first env build)
    try:
        sys.path.insert(0, str(LIBERO_PLUS_DIR))
        from libero_plus.benchmark import get_libero_plus_dict
        d = get_libero_plus_dict()
        print(f"[libero-plus] OK; dimensions available: {list(d.keys())}")
    except Exception as e:
        print(f"[libero-plus] WARN: import failed: {e}")


# ---------------------------------------------------------------------------
# Lang embedding precompute (small; do it now while we're here)
# ---------------------------------------------------------------------------

def precompute_lang_embeddings(suites: List[str], dim: int = 4096) -> None:
    """Walk every LIBERO BDDL task in the installed package, extract the
    natural-language instruction, and cache a zero-padded embedding under
    /workspace/data/lang_emb/<sha256(text)>.npy.

    For the v1 sprint we use a placeholder: zero-padded embedding of fixed
    shape. The real lang encoder (SigLIP text or Llama-7B) can be swapped in
    later — the datasets.py loader already keys by sha256(text), so swapping
    the embedding source is a one-script change.
    """
    section("Language instruction cache (placeholder embeddings)")
    try:
        from libero.libero import benchmark
    except ImportError:
        print("[lang] libero not importable; skipping")
        return
    import hashlib
    import numpy as np

    out_dir = ensure(DATA_ROOT / "lang_emb")
    instructions = set()
    for suite in suites:
        try:
            bench = benchmark.get_benchmark_dict()[suite]()
            for tid in range(bench.n_tasks):
                t = bench.get_task(tid)
                instructions.add(t.language.strip())
        except Exception as e:
            print(f"[lang] {suite}: {e}")

    n_written = 0
    for instr in instructions:
        h = hashlib.sha256(instr.encode()).hexdigest()[:16]
        p = out_dir / f"{h}.npy"
        if p.exists():
            continue
        # 32 zero tokens × dim — placeholder until real encoder is wired up
        arr = np.zeros((32, dim), dtype=np.float32)
        np.save(p, arr)
        n_written += 1
    print(f"[lang] cached {n_written} new (of {len(instructions)} unique) instructions to {out_dir}")
    print(f"[lang] NOTE: these are placeholder zeros. Replace before final paper runs.")


# ---------------------------------------------------------------------------
# Top-level
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true",
                    help="run libero + libero-pkg + models + libero-plus + lang in order")
    ap.add_argument("--libero", action="store_true", help="download LIBERO HDF5 demos")
    ap.add_argument("--libero-pkg", action="store_true",
                    help="clone + pip install the LIBERO sim package")
    ap.add_argument("--models", action="store_true",
                    help="cache V-JEPA-2 + OpenVLA-OFT model weights")
    ap.add_argument("--libero-plus", action="store_true",
                    help="clone + install LIBERO-Plus for perturbation eval")
    ap.add_argument("--lang-emb", action="store_true",
                    help="precompute placeholder language embeddings")

    ap.add_argument("--openvla-modified", action="store_true",
                    help="use clip-rt/modified_libero_hdf5 instead of yifengzhu mirror")
    ap.add_argument("--libero-suites", nargs="*", default=LIBERO_SUITES_V1,
                    help=f"suites to download (default: {LIBERO_SUITES_V1})")
    ap.add_argument("--openvla-suites", nargs="*", default=["spatial"],
                    help='which OFT checkpoints to cache (default: ["spatial"])')
    ap.add_argument("--openvla-combined", action="store_true",
                    help="also cache the 4-suite combined OFT checkpoint")
    ap.add_argument("--reinstall", action="store_true",
                    help="re-clone + reinstall libero / libero-plus")
    args = ap.parse_args()

    if not any([args.all, args.libero, args.libero_pkg, args.models,
                args.libero_plus, args.lang_emb]):
        ap.error("specify at least one of --all / --libero / --libero-pkg / "
                 "--models / --libero-plus / --lang-emb")

    section("Phase 1 download script")
    print(f"workspace: {WORKSPACE}")
    print(f"free space: {free_gb(WORKSPACE):.1f} GB")
    check_hf_login()

    ensure(WORKSPACE)
    ensure(DATA_ROOT)
    ensure(HF_CACHE)
    ensure(LOG_ROOT)

    t_start = time.time()

    if args.all or args.libero:
        download_libero(
            suites=args.libero_suites,
            use_openvla_modified=args.openvla_modified,
        )

    if args.all or args.libero_pkg:
        install_libero_package(reinstall=args.reinstall)

    if args.all or args.models:
        download_models(
            vjepa2=True,
            openvla_oft_suites=args.openvla_suites,
            openvla_combined=args.openvla_combined,
        )

    if args.all or args.libero_plus:
        install_libero_plus(reinstall=args.reinstall)

    if args.all or args.lang_emb:
        # Need the libero package to enumerate task instructions
        precompute_lang_embeddings(args.libero_suites)

    section(f"DONE in {t_human(time.time() - t_start)}")
    print(f"final free space: {free_gb(WORKSPACE):.1f} GB")
    print()
    print("Next step:")
    print("  python scripts/precompute_embeddings.py \\")
    print(f"      --input {LIBERO_ROOT} \\")
    print(f"      --output {DATA_ROOT}/embeddings/libero \\")
    print("      --source libero")


if __name__ == "__main__":
    main()
