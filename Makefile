SHELL := /bin/sh
.DEFAULT_GOAL := render

NVCC ?= nvcc
PYTHON ?= python3
ARCH ?= sm_75
NVCCFLAGS ?= -std=c++17 -O3 -lineinfo -Xptxas=-v
VARIANTS ?= ray-tracing-brute ray-tracing-bvh ray-tracing-bvh-max-opt
SCENES ?= 10 100 1000 5000 5001
SCENE_DIR ?= test
BUILD_DIR ?= build
OUTPUT_DIR ?= rendered-images

BINARIES := $(addprefix $(BUILD_DIR)/,$(VARIANTS))
vpath %.cu src src/variants

.PHONY: all build render check-png
all: render
build: $(BINARIES)

$(BUILD_DIR)/%: %.cu Makefile
	@mkdir -p "$(BUILD_DIR)"
	$(NVCC) $(NVCCFLAGS) -arch=$(ARCH) "$<" -o "$@"

check-png:
	@$(PYTHON) -c 'from PIL import Image' || { echo 'Install Pillow: python3 -m pip install Pillow' >&2; exit 1; }

# Run sequentially, even with make -j, so GPU timings do not overlap.
# Each process gets a fresh working directory for its fixed output filenames.
render: check-png build
	@set -eu; \
	for scene in $(SCENES); do \
		test -f "$(SCENE_DIR)/raytracer_scene_$$scene.txt" || { echo "Missing scene: $$scene" >&2; exit 1; }; \
	done; \
	mkdir -p "$(OUTPUT_DIR)/logs"; \
	work=$$(mktemp -d); \
	trap 'rm -rf "$$work"' 0; \
	trap 'exit 1' 1 2 3 15; \
	for variant in $(VARIANTS); do \
		for scene in $(SCENES); do \
			name="$$variant-scene-$$scene"; \
			mkdir "$$work/$$name"; \
			echo "Rendering $$name"; \
			if ! (cd "$$work/$$name" && "$(abspath $(BUILD_DIR))/$$variant" < "$(abspath $(SCENE_DIR))/raytracer_scene_$$scene.txt") > "$(OUTPUT_DIR)/logs/$$name.log" 2>&1; then \
				cat "$(OUTPUT_DIR)/logs/$$name.log" >&2; exit 1; \
			fi; \
			$(PYTHON) -c 'from PIL import Image; import sys; im = Image.open(sys.argv[1]); im.save(sys.argv[2], format="PNG")' "$$work/$$name/image.ppm" "$(OUTPUT_DIR)/$$name.png"; \
			for timing in "$$work/$$name/"*.txt; do \
				if test -f "$$timing"; then cp "$$timing" "$(OUTPUT_DIR)/logs/$$name-$${timing##*/}"; fi; \
			done; \
		done; \
	done
