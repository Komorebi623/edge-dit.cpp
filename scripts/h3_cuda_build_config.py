#!/usr/bin/env python3
"""Plan/validate/build H3 with local fixed dependencies; never launch inference.

All output, including compiler temporary files, belongs to a NEW H3_BUILD_ROOT
under this checkout's _scratch. Failed runs are retained, never cleaned up.
"""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile
import uuid

BASELINE = "97cc585d9dbd9ac5e30b78ef2e2c4a0d8bfbe9f2"
PINS = {
    "ggml": "f270346d688c7bec7c56bae2ae73343aba6817ad",
    "onednn": "5e9224036021433d2577548ed0539fe9a53256bc",
    "cudnn_frontend": "07b368a2696f5fd0fc2207d4635c51cea4c0bf71",
}
METHODS = "ED_H3_FUSED_OPS,ED_H3_CACHE_DIT,ED_H3_ADALN_CACHE,ED_H3_SAGE_ATTN,ED_H3_SPARSE_ATTN"


def absolute(value, root):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else root / path).resolve()


def executable(value):
    found = shutil.which(value)
    return str(Path(found).resolve()) if found else value


def config_from_env(root, mode, env):
    root = root.resolve()
    arch = env.get("H3_CUDA_ARCHITECTURES", "")
    # Do not accept native/all/OFF: these permit host-dependent autodetection.
    if not arch or not all(re.fullmatch(r"[1-9][0-9]{1,2}(?:a|f)?(?:-real|-virtual)?", a)
                           for a in arch.split(";")):
        raise ValueError("Set H3_CUDA_ARCHITECTURES explicitly (e.g. 90 or '90-real;90-virtual'); no GPU autodetection")
    if env.get("H3_BASELINE_REF", BASELINE) != BASELINE:
        raise ValueError("H3_BASELINE_REF must remain the fixed research baseline " + BASELINE)
    generator = env.get("H3_CMAKE_GENERATOR", "Ninja")
    if generator not in ("Ninja", "Unix Makefiles"):
        raise ValueError("H3_CMAKE_GENERATOR must be Ninja or Unix Makefiles")
    make_program = executable("ninja" if generator == "Ninja" else "make")
    jobs = env.get("H3_BUILD_JOBS", "4")
    if not re.fullmatch(r"[1-9][0-9]*", jobs):
        raise ValueError("H3_BUILD_JOBS must be a positive integer")
    output_value = env.get("H3_BUILD_ROOT", "_scratch/h3-cuda-" + uuid.uuid4().hex)
    raw_output = Path(output_value).expanduser()
    if not raw_output.is_absolute():
        raw_output = root / raw_output
    if raw_output.is_symlink():
        raise ValueError("Refusing a symlink build/output path: " + str(raw_output))
    output = absolute(output_value, root)
    scratch = root / "_scratch"
    # Resolve first, so a symlink cannot redirect writes outside our scratch.
    if scratch.is_symlink() or scratch not in output.parents:
        raise ValueError("H3_BUILD_ROOT must be a new directory inside " + str(scratch))
    missing = []

    def required(name, alias=None):
        value = env.get(name) or (env.get(alias) if alias else None)
        if not value:
            missing.append(name)
            return None
        # CMake represents lists with semicolons; paths must remain single values.
        if any(c in value for c in ";\n\r"):
            raise ValueError(name + " must be a single path, not a CMake list")
        return str(absolute(value, root))

    cuda = required("H3_CUDA_ROOT", "CUDAToolkit_ROOT")
    frontend = required("H3_CUDNN_FRONTEND_SOURCE")
    cudnn = required("H3_CUDNN_ROOT", "CUDNN_ROOT")
    c = executable(env.get("H3_C_COMPILER") or env.get("CC") or "cc")
    cxx = executable(env.get("H3_CXX_COMPILER") or env.get("CXX") or "c++")
    nvcc = str(Path(cuda) / "bin/nvcc") if cuda else "<H3_CUDA_ROOT>/bin/nvcc"
    include = str(Path(cudnn) / "include") if cudnn else "<H3_CUDNN_ROOT>/include"
    libraries = [Path(cudnn) / d / name for d in ("lib", "lib64")
                 for name in ("libcudnn.so", "libcudnn.so.9", "libcudnn.so.8")] if cudnn else []
    library = next((str(p.resolve()) for p in libraries if p.is_file()),
                   str(libraries[0]) if libraries else "<H3_CUDNN_ROOT>/lib/libcudnn.so")
    # The fixed frontend's finder uses lower-case cudnn_LIBRARY (not
    # CUDNN_LIBRARY). Pin every installed component as well, preventing its
    # find_library calls from accidentally selecting a second cuDNN install.
    cudnn_libraries = {"cudnn_LIBRARY": library}
    if cudnn:
        for folder in ("lib", "lib64"):
            for path in sorted((Path(cudnn) / folder).glob("libcudnn*.so*")):
                match = re.fullmatch(r"lib(cudnn(?:_[a-z_]+)?)\.so(?:\.[0-9.]+)?", path.name)
                if match and path.is_file():
                    cudnn_libraries.setdefault(match[1] + "_LIBRARY", str(path.resolve()))
    runtime = env.get("H3_CUDNN_RUNTIME_LIBRARY_DIRS", "")
    runtime_dirs = []
    if runtime:
        if any(not p or "\n" in p or "\r" in p for p in runtime.split(";")):
            raise ValueError("H3_CUDNN_RUNTIME_LIBRARY_DIRS must contain nonempty paths separated by semicolons")
        runtime_dirs = [str(absolute(p, root)) for p in runtime.split(";")]
    common = ["-G", generator, "-DCMAKE_BUILD_TYPE=Release",
              "-DCMAKE_MAKE_PROGRAM=" + make_program,
              "-DCMAKE_CUDA_ARCHITECTURES=" + arch,
              "-DCUDAToolkit_ROOT=" + (cuda or "<H3_CUDA_ROOT>"),
              "-DCMAKE_CUDA_COMPILER=" + nvcc,
              "-DCMAKE_C_COMPILER=" + c, "-DCMAKE_CXX_COMPILER=" + cxx,
              "-DCMAKE_CUDA_HOST_COMPILER=" + cxx,
              "-DCUDNN_ROOT=" + (cudnn or "<H3_CUDNN_ROOT>"),
              "-DCUDNN_INCLUDE_DIR=" + include, "-DCUDNN_STATIC=OFF",
              "-DCUDNN_RUNTIME_LIBRARY_DIRS=" + ";".join(runtime_dirs),
              "-DED_GGML_CUDA=ON", "-DED_ENABLE_CUDNN_SDPA=ON",
              # Existing CMake takes its FetchContent branch, but source override
              # + disconnected flags ensure it consumes ONLY the supplied tree.
              "-DED_FETCH_CUDNN_FRONTEND=ON",
              "-DFETCHCONTENT_SOURCE_DIR_CUDNN_FRONTEND=" + (frontend or "<H3_CUDNN_FRONTEND_SOURCE>"),
              "-DFETCHCONTENT_FULLY_DISCONNECTED=ON", "-DFETCHCONTENT_UPDATES_DISCONNECTED=ON",
              "-DGGML_CUDA_FA_ALL_QUANTS=OFF", "-DGGML_CCACHE=OFF",
              "-DGGML_CUDA_CUB_3DOT2=OFF"]
    common.extend("-D" + key + "=" + value for key, value in sorted(cudnn_libraries.items()))
    cmake = executable("cmake")
    phases = []
    for phase in ("baseline", "candidate"):
        if mode not in (phase, "all"):
            continue
        source = output / "baseline-source" if phase == "baseline" else root
        destination = output / phase
        sage = "OFF" if phase == "baseline" else "ON"
        phases.append({"phase": phase, "source": str(source), "destination": str(destination),
                       "sage_attention": sage,
                       "configure": [cmake, "-S", str(source), "-B", str(destination)] + common +
                                    ["-DED_ENABLE_CUDA_SAGE_ATTN=" + sage],
                       "build": [cmake, "--build", str(destination), "--target", "ed-cli", "-j", jobs]})
    return {"repo_root": str(root), "mode": mode, "baseline_revision": BASELINE,
            "dependency_pins": dict(PINS), "cuda_architectures": arch,
            "cmake_generator": generator, "make_program": make_program,
            "cuda_root": cuda, "cuda_compiler": nvcc, "c_compiler": c, "cxx_compiler": cxx,
            "cudnn_root": cudnn, "cudnn_include": include, "cudnn_library": library,
            "cudnn_libraries": cudnn_libraries,
            "cudnn_frontend_source": frontend, "runtime_library_dirs": runtime_dirs,
            "build_root": str(output), "common_cmake_args": common,
            "environment": {"TMPDIR": str(output / "tmp"), "TMP": str(output / "tmp"),
                            "TEMP": str(output / "tmp"), "CUDA_VISIBLE_DEVICES": "",
                            "CUDA_CACHE_PATH": str(output / "tmp/cuda-cache"),
                            "CCACHE_DISABLE": "1", "GIT_OPTIONAL_LOCKS": "0"},
            "missing_settings": missing, "phases": phases}


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args],
                                   env=dict(os.environ, GIT_OPTIONAL_LOCKS="0"), text=True).rstrip("\n")


def clean_pin(path, revision):
    path = Path(path)
    if not (path / "CMakeLists.txt").is_file():
        raise ValueError("Missing local dependency CMakeLists.txt: " + str(path))
    # An uninitialized submodule must not inherit the parent repository's HEAD.
    if Path(git(path, "rev-parse", "--show-toplevel")).resolve() != path.resolve():
        raise ValueError("Dependency must be its own Git checkout: " + str(path))
    if git(path, "rev-parse", "HEAD") != revision or git(path, "status", "--porcelain", "--untracked-files=all"):
        raise ValueError("Refusing changed/unpinned dependency: " + str(path) + " (requires " + revision + ")")
    # No recursive initialization/fetch: nested submodules must already be ready.
    if any(line and line[0] != " " for line in git(path, "submodule", "status", "--recursive").splitlines()):
        raise ValueError("Dependency has an uninitialized or mismatched nested submodule: " + str(path))
    return revision


def dependency_state(config):
    root = Path(config["repo_root"])
    state = {}
    for name in ("ggml", "onednn"):
        expected = PINS[name]
        for ref in (BASELINE, "HEAD"):
            entry = git(root, "ls-tree", ref, "third_party/" + name).split()
            if len(entry) < 3 or entry[:3] != ["160000", "commit", expected]:
                raise ValueError("Research gitlink changed: " + ref + ":third_party/" + name)
        state[name] = clean_pin(root / "third_party" / name, expected)
    frontend = Path(config["cudnn_frontend_source"])
    state["cudnn_frontend"] = clean_pin(frontend, PINS["cudnn_frontend"])
    for name in ("cmake/cuDNN.cmake", "include/cudnn_frontend.h"):
        if not (frontend / name).is_file():
            raise ValueError("Incomplete cudnn-frontend: " + str(frontend / name))
    vendored = root / "third_party/cudnn-frontend"
    if (vendored / "CMakeLists.txt").exists() and vendored.resolve() != frontend:
        raise ValueError("Vendored cudnn-frontend would bypass the shared source override; select its exact path")
    # Lock the explicit native entry points too, not just the Git dependencies.
    # CUDA runtime/driver compatibility remains the main agent's audit.
    native = list(Path(config["cudnn_include"]).glob("cudnn*.h")) + [
              *map(Path, config["cudnn_libraries"].values()), Path(config["cuda_compiler"]),
              Path(config["c_compiler"]), Path(config["cxx_compiler"])]
    state["native_files_sha256"] = {str(p): file_hash(p) for p in native}
    state["candidate_revision"] = git(root, "rev-parse", "HEAD")
    return state


def validate(config):
    if config["missing_settings"]:
        raise ValueError("Missing explicit local settings: " + ", ".join(config["missing_settings"]))
    output = Path(config["build_root"])
    if output.exists() or output.is_symlink():
        raise ValueError("Refusing existing build/output path (including empty directories): " + str(output))
    for tool in (config["cuda_compiler"], config["c_compiler"], config["cxx_compiler"],
                 config["phases"][0]["configure"][0], config["make_program"]):
        if not Path(tool).is_file() or not os.access(tool, os.X_OK):
            raise ValueError("Missing executable tool (no installation attempted): " + tool)
    cuda = Path(config["cuda_root"])
    if not (cuda / "include/cuda.h").is_file():
        raise ValueError("Incomplete CUDA toolkit: " + str(cuda))
    cudnn = Path(config["cudnn_root"])
    for path in (Path(config["cudnn_include"]) / "cudnn.h",
                 Path(config["cudnn_include"]) / "cudnn_version.h", *map(Path, config["cudnn_libraries"].values())):
        if not path.is_file() or cudnn not in path.resolve().parents:
            raise ValueError("Missing or out-of-root cuDNN file: " + str(path))
    major_match = re.search(r"#define\s+CUDNN_MAJOR\s+([0-9]+)",
                            (Path(config["cudnn_include"]) / "cudnn_version.h").read_text())
    if not major_match or major_match[1] not in ("8", "9"):
        raise ValueError("Fixed frontend requires an explicitly verified cuDNN 8/9 installation")
    required = ("cudnn_graph", "cudnn_engines_runtime_compiled") if major_match[1] == "9" else (
        "cudnn_adv_infer", "cudnn_adv_train", "cudnn_cnn_infer", "cudnn_cnn_train", "cudnn_ops_infer", "cudnn_ops_train")
    if any(name + "_LIBRARY" not in config["cudnn_libraries"] for name in required):
        raise ValueError("Incomplete fixed cuDNN component libraries under " + str(cudnn))
    for path in config["runtime_library_dirs"]:
        if not Path(path).is_dir():
            raise ValueError("Missing runtime library directory: " + path)
    return dependency_state(config)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def snapshot_baseline(root, snapshot):
    # The destination is always freshly created; no reuse/repair of old snapshots.
    snapshot.mkdir()
    archive = subprocess.check_output(["git", "-C", str(root), "archive", BASELINE],
                                      env=dict(os.environ, GIT_OPTIONAL_LOCKS="0"))
    with tarfile.open(fileobj=io.BytesIO(archive)) as stream:
        for entry in stream:
            target = snapshot / entry.name
            if entry.name.startswith("/") or ".." in Path(entry.name).parts or entry.issym() or entry.islnk():
                raise ValueError("Unsafe entry in pinned baseline archive: " + entry.name)
            if entry.isdir():
                target.mkdir(parents=True, exist_ok=True)
            elif entry.isfile():
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open("xb") as out:
                    shutil.copyfileobj(stream.extractfile(entry), out)
                target.chmod(entry.mode)
            else:
                raise ValueError("Unexpected archive entry: " + entry.name)
    for name in ("ggml", "onednn"):
        destination = snapshot / "third_party" / name
        destination.mkdir(parents=True, exist_ok=True)
        for entry in (root / "third_party" / name).iterdir():
            if entry.name not in (".git", "build"):
                (destination / entry.name).symlink_to(entry, target_is_directory=entry.is_dir())


def check_cache(config, destination, sage):
    values = {}
    for line in (destination / "CMakeCache.txt").read_text().splitlines():
        if "=" in line and not line.startswith(("#", "//")):
            key, value = line.split("=", 1)
            values[key.split(":", 1)[0]] = value
    expected = dict(arg[2:].split("=", 1) for arg in config["common_cmake_args"] if arg.startswith("-D"))
    expected["ED_ENABLE_CUDA_SAGE_ATTN"] = sage
    # Check what CMake actually consumed, not merely what the script requested.
    for key, value in expected.items():
        actual = values.get(key)
        if key.endswith(("COMPILER", "_LIBRARY")) or key in ("CUDNN_INCLUDE_DIR", "CMAKE_MAKE_PROGRAM"):
            matches = actual is not None and Path(actual).resolve() == Path(value).resolve()
        else:
            matches = actual == value
        if not matches:
            raise ValueError("CMake configuration drift: " + key + " expected " + value + ", got " + str(actual))


def write_provenance(config, state, phase):
    source, destination = Path(phase["source"]), Path(phase["destination"])
    revision = BASELINE if phase["phase"] == "baseline" else state["candidate_revision"]
    build_config = destination / "build-config.txt"
    values = {"git_commit": revision, "ggml_commit": PINS["ggml"]}
    lines = build_config.read_text().splitlines()
    build_config.write_text("\n".join(line.split("=", 1)[0] + "=" + values[line.split("=", 1)[0]]
                                     if line.split("=", 1)[0] in values else line for line in lines) + "\n")
    entries = [{"path": p.relative_to(source).as_posix(), "sha256": file_hash(p)}
               for folder in ("src", "include") for p in sorted((source / folder).rglob("*")) if p.is_file()]
    entries.append({"path": "CMakeLists.txt", "sha256": file_hash(source / "CMakeLists.txt")})
    identity = hashlib.sha256(json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    with (destination / "h3-source-manifest.json").open("x") as stream:
        json.dump({"source_tree_sha256": identity, "files": entries}, stream, indent=2)
        stream.write("\n")
    diff = subprocess.check_output(["git", "-C", config["repo_root"], "diff", "--binary", "--", "src", "include", "CMakeLists.txt"],
                                   env=dict(os.environ, GIT_OPTIONAL_LOCKS="0"))
    provenance = {"source_revision": revision, "ggml_revision": PINS["ggml"], "onednn_revision": PINS["onednn"],
                  "cuda_architectures": config["cuda_architectures"], "cudnn_frontend_revision": PINS["cudnn_frontend"],
                  "cuda_sage_attention": phase["sage_attention"], "source_tree_sha256": identity,
                  "binary_sha256": file_hash(destination / "bin/ed-cli"),
                  "tracked_diff_sha256": hashlib.sha256(diff).hexdigest()}
    if phase["phase"] == "candidate":
        provenance["implemented_methods"] = METHODS
    # Every phase gets the exact SAME shared configuration and dependency stamp.
    shared = {"common_cmake_args": config["common_cmake_args"], "dependency_state": state,
              "environment": {k: config["environment"][k] for k in ("CUDA_VISIBLE_DEVICES", "CCACHE_DISABLE")}}
    encoded = json.dumps(shared, sort_keys=True, separators=(",", ":")).encode()
    provenance["shared_config_sha256"] = hashlib.sha256(encoded).hexdigest()
    with (destination / "h3-build-provenance.txt").open("x") as stream:
        stream.write("".join(key + "=" + value + "\n" for key, value in provenance.items()))
    with (destination / "h3-shared-build-config.json").open("x") as stream:
        json.dump(shared, stream, indent=2)
        stream.write("\n")


def build(config, state):
    root, output = Path(config["repo_root"]), Path(config["build_root"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir()  # Atomic refusal of an existing path, even after validation.
    Path(config["environment"]["TMPDIR"]).mkdir()
    env = dict(os.environ, **config["environment"])
    with (output / "h3-build-plan.json").open("x") as stream:
        json.dump(config, stream, indent=2)
        stream.write("\n")
    for phase in config["phases"]:
        if dependency_state(config) != state:
            raise ValueError("Dependency/toolchain/revision changed after validation")
        if phase["phase"] == "baseline":
            snapshot_baseline(root, Path(phase["source"]))
        destination = Path(phase["destination"])
        destination.mkdir()
        subprocess.run(phase["configure"], cwd=root, env=env, check=True)
        check_cache(config, destination, phase["sage_attention"])
        subprocess.run(phase["build"], cwd=root, env=env, check=True)
        if dependency_state(config) != state:
            raise ValueError("Dependency/toolchain/revision changed during build; do not use this output")
        write_provenance(config, state, phase)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="build_h3_cuda.sh", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Required explicit local settings:
  H3_CUDA_ARCHITECTURES       e.g. 90 for the confirmed H200; no default/native
  H3_CUDA_ROOT               existing toolkit root (or CUDAToolkit_ROOT)
  H3_CUDNN_ROOT              existing cuDNN root with include/ and lib/ or lib64/
                            (CUDNN_ROOT is also accepted)
  H3_CUDNN_FRONTEND_SOURCE   clean local checkout at the fixed 07b368a2696f... pin
Optional:
  H3_CMAKE_GENERATOR         Ninja (default) or Unix Makefiles; never installs a tool
  H3_BUILD_ROOT              NEW path within repo/_scratch (default: unique path)
  H3_C_COMPILER / H3_CXX_COMPILER   executable paths (default: CC/cc, CXX/c++)
  H3_BUILD_JOBS              positive integer, default 4
  H3_CUDNN_RUNTIME_LIBRARY_DIRS    fixed extra paths separated by semicolons
No downloads, dependency updates, output reuse, cleanup, or GPU execution.
--dry-run previews even if dependencies/settings are missing; it is NOT validation.
--validate checks local files/Git pins only, without CMake, nvcc, GPU, or writes.
The sole intentional phase option difference is Sage OFF (baseline) / ON (candidate).
""")
    parser.add_argument("mode", nargs="?", choices=("baseline", "candidate", "all"), default="all")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--dry-run", "--preview", action="store_true", dest="preview")
    action.add_argument("--validate", action="store_true")
    args = parser.parse_args(argv)
    try:
        config = config_from_env(Path(__file__).resolve().parents[1], args.mode, os.environ)
        if args.preview:
            print(json.dumps({"action": "preview", "validation_performed": False,
                              "note": "Paths/pins/compiler support are unverified; no commands executed or files written",
                              **config}, indent=2))
            return 0
        state = validate(config)
        if args.validate:
            print(json.dumps({"action": "validate", "validation_performed": True,
                              "note": "Local configuration only; compiler/driver/GPU compatibility not tested",
                              "dependency_state": state, **config}, indent=2))
            return 0
        print("New H3 build output (retained on failure): " + config["build_root"], flush=True)
        build(config, state)
        return 0
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        print("H3 CUDA build refused: " + str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
