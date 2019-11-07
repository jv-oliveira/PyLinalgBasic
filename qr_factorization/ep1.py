#!venv/bin/python

import getopt
import sys

import qr_factorization as qr
from matrix import *

input_file = 'input.csv'
data_type = None
delimiter = ','
max_iterations = 1000
epsilon = 1e-2
output_file = None


def usage():
    print('ep1.py -i <inputfile> -t <data_type> -d <delimiter> -e <epsilon> -m <max_iterations>'
          ' -o <outputfile>')


def print_result(output=sys.stdout):
    print("values:", values, file=output)
    print("vectors:\n", Matrix.transpose(vectors), file=output)


try:
    opts, args = getopt.getopt(sys.argv[1:],
                               "hi:t:d:e:m:o:",
                               ["ifile=",
                                "dtype=",
                                "delimiter=",
                                "epsilon=",
                                "max_iterations=",
                                "ofile="])
except getopt.GetoptError:
    usage()
    sys.exit(2)
for opt, arg in opts:
    if opt == '-h':
        usage()
        sys.exit()
    elif opt in ("-i", "--ifile"):
        input_file = arg
    elif opt in ("-t", "--dtype"):
        data_type = arg
    elif opt in ("-d", "--delimiter"):
        delimiter = arg
    elif opt in ("-e", "--epsilon"):
        epsilon = float(arg)
    elif opt in ("-m", "--max_iterations"):
        max_iterations = int(arg)
    elif opt in ("-o", "--ofile"):
        output_file = arg

if output_file is None:
    output = sys.stdout
else:
    output = open(output_file, "w")

print("Reading Matrix in", input_file)
print("Data type:", data_type)
print("Delimiter:", delimiter)

A = Matrix.from_file(input_file, data_type, delimiter)

values, vectors = qr.eigenvalues_and_eigenvectors(A, epsilon, max_iterations)

print("vectors norms:", [vectors[i].norm() for i in range(vectors.num_columns())])

print_result(output)

if output_file is not None:
    print_result()
    output.close()
