import numpy as np
import glob
from matplotlib.pyplot import imshow, savefig
import matplotlib.pyplot as plt

if __name__=="__main__":
    for file in glob.glob('*.txt'):
    array = np.loadtxt(file)
    fig = imshow(array)
    plt.colorbar()
    savefig(file.split('.')[0]+ '.png')
    plt.close()
