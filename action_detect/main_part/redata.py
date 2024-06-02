import os
import numpy as np
import simplejson
data_path= 'data_test/coordinates1'
from smt import show_joint
from features import get_coordinate

coordinate = get_coordinate(data_path)
print(coordinate)
