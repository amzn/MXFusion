# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SPHINXPROJ    = MXFusion
SOURCEDIR     = .
BUILDDIR      = _build
EXAMPLES_SYM_DIR      = examples
EXAMPLES_BASE_DIR     = ../examples

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	rm -rf _build generated examples || true

	# copy over the examples folder so notebooks get compiled into the docs
	cp -r $(EXAMPLES_BASE_DIR) $(EXAMPLES_SYM_DIR)

	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)


# @$(SPHINXAPIDOC) $(SPHINXAPIDOC_ARGS) -o "$(GENERATED_DIR)" "$(PROJECT_DIR)"  $(O)
