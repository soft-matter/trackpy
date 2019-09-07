NOTEBOOK_DIR = ../../../trackpy-examples/notebooks
FILES := $(patsubst $(NOTEBOOK_DIR)/%,%, $(wildcard $(NOTEBOOK_DIR)/*.ipynb))

notebooks:
	# run the nbconvert and output an html to current directory
	# write an .rst file so that sphinx will include the .html
	# the first line is a title, which is necessary for it to be included in the Sphinx toctree
	@- $(foreach FILE, $(FILES), \
	    jupyter nbconvert --to html $(NOTEBOOK_DIR)/$(FILE) --output $(CURDIR)/$(FILE:.ipynb=.html); \
	    bash -c 'echo -e "$(FILE:.ipynb=)\n------------------------------------------------\n.. raw:: html\n    :file: "$(FILE:.ipynb=.html)' > $(FILE:.ipynb=.rst); \
	)

