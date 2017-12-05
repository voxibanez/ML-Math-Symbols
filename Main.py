#Call gui and build neural network in seperate threads

#Call GUI

from inkml_Interop import *
from train_network import *
import sys
#parseItem("Test/dif_eqn15.inkml")
if len(sys.argv) == 2:
    start_training(sys.argv[1])
    print 'Hey I actually finished training'
else:
    print 'Usage: python Main.py training_data_directory'
