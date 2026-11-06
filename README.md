# CUDA Ray Tracer — CS23B002

CUDA implementations of brute-force ray tracing and bounding volume hierarchy
(BVH) traversal. Source code is in `src/`, individual optimization experiments
are in `src/variants/`, and scene inputs are in `test/`.

## Lightweight download (recommended)

To download only the CUDA sources, Makefile and this README, use a partial clone
with sparse checkout:

```sh
git clone --depth 1 --filter=blob:none --no-checkout \
  https://github.com/MrKaushik7/Ray-Tracer-Course-Project.git
cd Ray-Tracer-Course-Project
git sparse-checkout set --no-cone '/src/*.cu' '/src/variants/*.cu' '/Makefile' '/README.md'
git checkout
make build
```

This fetches the latest commit's metadata and the selected file contents,
leaving images, reports, results and scene files undownloaded. The explicit
file patterns also exclude large files at the repository root. The repository
cannot enforce this selection for a normal `git clone`; readers must use the
commands above. `.gitignore` does not prevent tracked files from downloading.

Fetch scenes when ready to render:

```sh
git sparse-checkout add '/test/'
python3 -m pip install Pillow
make
```

Fetch optional directories when needed (choose any of these):

```sh
git sparse-checkout add '/scripts/'                  # benchmark/analysis scripts
git sparse-checkout add '/benchmarks/'               # earlier text measurements
git sparse-checkout add '/results/' '/constant-scene-results/'
git sparse-checkout add '/images/'                   # saved renders and graphs
```

Only files committed and pushed to GitHub are available through these commands.
Use `git pull --ff-only --no-stat` for updates; the sparse selection stays in
place. `--no-stat` avoids fetching excluded contents to calculate the change
summary. Git may fetch additional contents if another command needs them. To fetch and
check out every file at the current revision later:

```sh
git sparse-checkout disable
```

The clone above omits older history; `git fetch --unshallow` retrieves that
history if needed. Existing full clones have already downloaded their contents;
sparse checkout cannot undo that download. See Git's documentation for
[partial clones](https://git-scm.com/docs/git-clone) and
[sparse checkout](https://git-scm.com/docs/git-sparse-checkout).

## Compile and render

Requirements: an NVIDIA GPU, a compatible CUDA toolkit with `nvcc` on PATH,
GNU Make, Python 3, and Pillow for PNG conversion. Run from the repository root:

```sh
python3 -m pip install Pillow
make
```

This compiles **only** these three implementations and runs each on scenes
**10, 100, 1000, 5000, and 5001**:

- `src/ray-tracing-brute.cu`: baseline brute force.
- `src/ray-tracing-bvh.cu`: baseline BVH.
- `src/ray-tracing-bvh-max-opt.cu`: combined BVH optimizations.

Executables go into `build/`. The 15 PNGs go into `rendered-images/`, for example
`rendered-images/ray-tracing-bvh-max-opt-scene-5001.png`. Console logs and the
programs' kernel timing files go into `rendered-images/logs/`. The programs run
sequentially in temporary directories; their PPM output is converted losslessly
to PNG and the temporary files are removed. Re-running `make` regenerates the
selected images and replaces their logs.

Compilation uses:

```sh
nvcc -std=c++17 -O3 -lineinfo -Xptxas=-v -arch=sm_75 SOURCE.cu -o EXECUTABLE
```

`sm_75` targets the Tesla T4 used for the experiments. Set `ARCH` to match your
GPU. Use `-B` when changing flags or architecture to force recompilation:

```sh
make -B ARCH=sm_86                 # example: an sm_86 GPU
make build                       # compile the three programs without rendering
make SCENES="5000 5001"           # render just the two large scenes
```

The current sources render at 800 × 600. Resolution, reflection depth and block
dimensions are source constants, not command-line arguments. Scene 5001 has a
different arrangement from scene 5000; it is a separate workload.

## Run another variant

Use the source filename without `.cu`. The Makefile searches both `src/` and
`src/variants/`:

```sh
make VARIANTS=ray-tracing-bvh-aabb-axis-split SCENES="5000 5001"
make VARIANTS=ray-tracing-brute-sphere-arithmetic SCENES=1000
```

To choose another output folder, append `OUTPUT_DIR=my-renders`. For example,
the first command creates `rendered-images/ray-tracing-bvh-aabb-axis-split-scene-5000.png`
and the corresponding scene-5001 PNG.

Constant-memory variants support at most 4000 spheres. Use the provided prefix
scenes, rather than the default large scenes:

```sh
nvcc -std=c++17 -O3 -lineinfo -Xptxas=-v -arch=sm_75 \
  src/variants/ray-tracing-bvh-constant-sphereAttr.cu -o /tmp/bvh-constant
mkdir -p constant-render
(cd constant-render && /tmp/bvh-constant < ../test/constant-prefixes/prefix5000_4000.txt)
python3 -c 'from PIL import Image; Image.open("constant-render/image.ppm").save("constant-render/image.png")'
rm constant-render/image.ppm
```

For any manual run, supply the scene on standard input: `./EXECUTABLE < SCENE.txt`.
The executable writes `image.ppm` and a timing text file in its working directory;
use Pillow as above to obtain a PNG. Run in separate directories to avoid
overwriting another image or appending to an earlier timing file.

## Benchmarks and profiling

Each current renderer performs **5 warmup kernel launches**, then uses CUDA
events to time **20 launches** and reports their average in milliseconds. This
measures GPU rendering work, excluding CPU BVH construction, data transfers,
image writing and PNG conversion. `make` performs one process run per selected
implementation/scene; use the benchmark scripts for repeated measurements.

```sh
# Benchmark the three top-level sources on all test/*.txt scenes, without NCU
# or fast math. Three process repetitions per experiment by default.
python3 scripts/run_colab_benchmarks.py --skip-ncu --skip-fast-math

# Also benchmark fast math separately and collect NCU profiles on scene 5000.
python3 scripts/run_colab_benchmarks.py

# Constant-memory prefix experiments and the separate large-scene comparison.
python3 scripts/run_constant_scene_experiments.py
```

The general runner discovers top-level `src/*.cu`; it does not recurse into
`src/variants/`. To benchmark a different set, copy the desired `.cu` files into
a separate directory and pass `--src-dir PATH`. Constant-memory sources require
the smaller prefix scenes; use the dedicated constant-scene runner for those.
The scripts detect GPU architecture, or accept `--arch 75` for T4. Nsight Compute
(`ncu`) and permission to access GPU performance counters are required for
profiling; `--skip-ncu` runs benchmarks without profiling.

Results are stored as follows:

- `benchmarks/`: earlier text measurements for block sizes, leaf sizes,
  reflection depth and resolution. Files named `retry` are separate reruns.
- `results/<timestamp>/`: general benchmark runs, including `raw.csv`
  (individual process averages), `summary.csv` (statistics across repetitions),
  `fast-math-comparison.csv` when applicable, build/run logs, and profiles.
  Historical runs may include more variants than the current top-level sources.
- `constant-scene-results/`: saved constant-memory and scene-comparison results.
  The dedicated runner can create a fresh output directory; see its `--help`.
- `axis-retest-results.zip`: the focused axis-split retest, with 30 paired runs
  per scene (5000 and 5001), using 20 warmups and 100 timed launches per run.
- `images/`: existing rendered outputs and analysis graphs.

For the general runner, each summary sample is a process-level average of 20
launches. By default there are three samples per experiment. NCU profiling is
separate: it skips the five warmups and profiles **one kernel launch**, possibly
replaying it to collect counters. Its timings are excluded from benchmark
summaries. Use unprofiled CUDA-event measurements for runtime comparisons.
Input hashes, compiler flags, GPU details and commands are recorded with the
script results. Keep hardware and flags consistent when comparing variants.
