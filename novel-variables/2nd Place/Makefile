.PHONY: clean data lint requirements create_environment help sync_data_to_s3 sync_data_from_s3

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
PROJECT_NAME = example_submission
PYTHON_INTERPRETER = python3
VENV_DIR = .venv

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Install Python Dependencies
requirements: create_environment
	@echo ">>> Installing dependencies from requirements.txt"
	$(VENV_DIR)/bin/pip install -r requirements.txt

## Delete all compiled Python files
clean:
	@echo ">>> Cleaning compiled Python files and cache"
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf $(VENV_DIR)

## Lint using flake8
lint: create_environment
	@echo ">>> Linting code with flake8"
	$(VENV_DIR)/bin/flake8 src

## Set up python3 virtual environment
create_environment:
	@echo ">>> Setting up Python virtual environment in $(VENV_DIR)"
	@if [ ! -d "$(VENV_DIR)" ]; then \
		$(PYTHON_INTERPRETER) -m venv $(VENV_DIR); \
		$(VENV_DIR)/bin/pip install --upgrade pip; \
		@echo ">>> Virtual environment created. Activate with:\nsource $(VENV_DIR)/bin/activate"; \
	else \
		@echo ">>> Virtual environment already exists. Skipping creation."; \
	fi

#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

## Show available commands
help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')

