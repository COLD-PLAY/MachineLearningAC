import numpy as np

randMat = np.mat(np.random.rand(4, 4))
invRandMat = randMat.I
tranRandMat = randMat.T

I = invRandMat * randMat

print(np.eye(4))