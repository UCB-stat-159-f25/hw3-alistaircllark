# ====================================================================
# Makefile for LOSC Computational Data Project
# Targets for environment setup, documentation build, and cleanup.
# ====================================================================

# Configuration Variables
# Environment name (used by conda/mamba)
CONDA_ENV_NAME = ligo

# Targets that don't produce a file (recommended practice)
.PHONY: env html clean

# --------------------------------------------------------------------
# 1. env: Creates or updates the Conda environment (losc-env)
# Using '-f environment.yml --prune' updates the existing environment 
# or creates it if it doesn't exist.
# --------------------------------------------------------------------
env: environment.yml
	@echo "--- Updating or creating $(CONDA_ENV_NAME) environment ---"
	@conda env update -f environment.yml --prune
	@echo "--- Environment setup complete. Run 'conda activate $(CONDA_ENV_NAME)' ---"

# --------------------------------------------------------------------
# 2. html: Builds the MyST site documentation
# We use 'conda run' to ensure the 'myst' command is executed 
# from the correct environment.
# --------------------------------------------------------------------
html:
	@echo "--- Building MyST documentation into HTML ---"
	@conda run -n $(CONDA_ENV_NAME) myst build --html
	@echo "--- HTML documentation built successfully in _build/html ---"

# --------------------------------------------------------------------
# 3. clean: Removes build and output directories/files
# --------------------------------------------------------------------
clean:
	@echo "--- Cleaning up figures, audio, and _build folders ---"
	@rm -rf figures audio _build
	@echo "--- Cleanup complete ---"