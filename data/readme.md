# How to download the Materials Project CIF files

In order to obtain the CIF file of the MP database, you need to apply for an "API_KEY".

1. Obtain an API key from the Materials Project database [[link](https://www.materialsproject.org)];
2. Replace the "API_KEY" placeholder in CIF_grabber.py with YOUR API KEY;
3. Ensure pymatgen is installed (e.g. `pip install pymatgen`);
4. Run the script CIF_grabber.py.

This will place all downloaded CIF files in the cifs/ folder. These can then be used by our model. Note that the Materials Project database is contantly updating, so the structures and properties may have changed since the publication of our paper.
