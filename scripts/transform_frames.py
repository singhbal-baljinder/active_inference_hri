from scipy.spatial.transform import Rotation as R
b_R_l = R.from_euler('zyx', [3.14, -0.64, 3.14], degrees=False)
l_R_s = R.from_euler('zyx', [-1.57, -1.57, -1.57], degrees=False)
b_R_s = b_R_l.as_matrix()@l_R_s.as_matrix()
print("b_R_l", b_R_l.as_matrix())  

