import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ecoscape_connectivity import SingleIterator

alist = [1, 2, 3, 4, 5]
it = SingleIterator(iter(alist))
for a, b in it:
    print(a, b)

it = SingleIterator(alist)
for a, b in it:
    print(a, b)
