import numpy as np
import pandas as pd
from ellipsoid_fit import ellipsoid_fit as ellipsoid_fit, data_regularize


if __name__ == '__main__':

    df = pd.read_csv("double-magnetometer-calibration.csv", header=0)

    xyzs = np.array([df['X1'], df['Y1'], df['Z1']]).transpose()

    center, evecs, radii, v = ellipsoid_fit(xyzs)

    a, b, c = radii
    r = (a * b * c) ** (1. / 3.)
    D = np.array([[r/a, 0., 0.], [0., r/b, 0.], [0., 0., r/c]])
    transformation = evecs.dot(D).dot(evecs.T)
    
    print('')
    print('center: ', center)
    print('radii: ', radii)
    print('evecs: ', evecs)
    print('transformation:')
    print(transformation)
    print('Coefficients:')
    print(v)
    
    np.savetxt('magcal_ellipsoid.txt', np.vstack((center.T, transformation)))

