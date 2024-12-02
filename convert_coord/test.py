import numpy as np

cameraMtx = np.load("Original camera matrix.npy")
rvec = np.load("RVec.npy")
tvec = np.load("TVec.npy")

Cc = np.matrix([0, 0, 0]).T
Pc = np.matrix([0, 0, 1]).T     # u = 300, v = 400
Pw = rvec.T * (Pc - tvec)
Cw = rvec.T * (Cc - tvec)

k = Cw[2]/(Cw[2]-Pw[2])
P = Cw + (Pw-Cw) * k

# P = np.matrix([0, 0, 0])
print("P:\n", P)

Pxyz1 = np.insert(P, 3, 1).reshape(4, 1)

print(Pxyz1)

xy = cameraMtx * (np.append(rvec, tvec, axis=1) * Pxyz1)
xy = xy / xy[2]

# camera matrix
# fx 0 cx
# 0 fy cy
# 0  0  1

u = (xy[0]-cameraMtx[0][2])/cameraMtx[0][0]
v = (xy[1]-cameraMtx[1][2])/cameraMtx[1][1]

print("u\n", u)
print("v\n", v)