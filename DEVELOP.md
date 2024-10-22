## For Developers

To build locally: 

    python3 -m build

To upload: 

    python3 -m twine upload dist/*

To build the documentation:

    pydoc-markdown -I ecoscape_connectivity -m repopulation -m distributions -m util > Documentation.md
    cat README-prototype.md Documentation.md > README.md

To install locally: 

    python3 -m pip install . 
