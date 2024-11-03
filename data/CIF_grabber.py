import csv
from mp_api.client import MPRester
from pymatgen.io.cif import CifFile
from pymatgen.io.cif import CifWriter
import pandas as pd

API_KEY = "API_KEY"
no_download_id = []


def getMaterialsInFile(csvfile):
    with MPRester(API_KEY) as m:
        k = 0
        for _id in csvfile:
            _id = _id[0]
            struct = m.get_structure_by_material_id(_id)
            k = k + 1
            print("---------------------k-th---------------------")
            try:
                file = CifWriter(struct, symprec=1.0)
                file.write_file(f'cifs/{_id}.cif')
            except:
                no_download_id.append(_id)
            if k == 153235:
                break

        data2 = pd.DataFrame(data=no_download_id)
        data2.to_csv("no_download_id.csv")


def main():
    file_names = ['cifs/id_prop.csv']
    for file in file_names:
        with open(file, 'r', newline='') as csvfile:
            csvreader = csv.reader(csvfile)
            getMaterialsInFile(csvreader)


if __name__ == '__main__':
    main()
