import numpy as np

# ดัชนี BlazePose (33 จุด)
NOSE=0
L_EYE_IN=1; L_EYE=2; L_EYE_OUT=3
R_EYE_IN=4; R_EYE=5; R_EYE_OUT=6
L_EAR=7; R_EAR=8
MOUTH_L=9; MOUTH_R=10
L_SHOULDER=11; R_SHOULDER=12
L_ELBOW=13;   R_ELBOW=14
L_WRIST=15;   R_WRIST=16
L_PINKY=17;   R_PINKY=18
L_INDEX=19;   R_INDEX=20
L_THUMB=21;   R_THUMB=22
L_HIP=23;     R_HIP=24
L_KNEE=25;    R_KNEE=26
L_ANKLE=27;   R_ANKLE=28
L_HEEL=29;    R_HEEL=30
L_FIDX=31;    R_FIDX=32

def _pair_midpoint(a, b):
    return (a + b) / 2.0

def _dist(a, b):
    return np.linalg.norm(a - b) + 1e-8

def _angle_deg(a, b, c):
    """มุมที่จุด b จากเวกเตอร์ BA และ BC (หน่วยองศา)"""
    ba = a - b
    bc = c - b
    denom = (np.linalg.norm(ba) * np.linalg.norm(bc)) + 1e-8
    cosang = np.clip(np.dot(ba, bc) / denom, -1.0, 1.0)
    return np.degrees(np.arccos(cosang))

def _joint_angles(xyz):
    """คำนวณมุมข้อต่อ (ใช้พอประมาณ 12 มุม) บนพิกัด normalized"""
    A = []
    # แขน
    A.append(_angle_deg(xyz[L_SHOULDER], xyz[L_ELBOW], xyz[L_WRIST]))
    A.append(_angle_deg(xyz[R_SHOULDER], xyz[R_ELBOW], xyz[R_WRIST]))
    # ไหล่ (torso-shoulder-elbow)
    A.append(_angle_deg(xyz[L_HIP], xyz[L_SHOULDER], xyz[L_ELBOW]))
    A.append(_angle_deg(xyz[R_HIP], xyz[R_SHOULDER], xyz[R_ELBOW]))
    # ขา (สะโพก-เข่า-ข้อเท้า)
    A.append(_angle_deg(xyz[L_HIP], xyz[L_KNEE], xyz[L_ANKLE]))
    A.append(_angle_deg(xyz[R_HIP], xyz[R_KNEE], xyz[R_ANKLE]))
    # สะโพก (ไหล่-สะโพก-เข่า)
    A.append(_angle_deg(xyz[L_SHOULDER], xyz[L_HIP], xyz[L_KNEE]))
    A.append(_angle_deg(xyz[R_SHOULDER], xyz[R_HIP], xyz[R_KNEE]))
    # เอียงลำตัว (ไหล่ซ้าย-สะโพกกลาง-ไหล่ขวา & สะโพกซ้าย-ไหล่กลาง-สะโพกขวา)
    hip_c = _pair_midpoint(xyz[L_HIP], xyz[R_HIP])
    sh_c  = _pair_midpoint(xyz[L_SHOULDER], xyz[R_SHOULDER])
    A.append(_angle_deg(xyz[L_SHOULDER], hip_c, xyz[R_SHOULDER]))
    A.append(_angle_deg(xyz[L_HIP],     sh_c,  xyz[R_HIP]))
    # เข่า-ข้อเท้า-ปลายเท้า
    A.append(_angle_deg(xyz[L_KNEE], xyz[L_ANKLE], xyz[L_FIDX]))
    A.append(_angle_deg(xyz[R_KNEE], xyz[R_ANKLE], xyz[R_FIDX]))
    return np.array(A, dtype=np.float32)

def landmarks_to_feature(landmarks):
    """
    landmarks: list[(x, y, z, visibility)] ยาว 33
    return: 1D feature vector: [xyz_norm(33*3), visibility(33), angles(12)] = 141 มิติ
    """
    lm = np.array(landmarks, dtype=np.float32)  # (33,4)
    xyz = lm[:, :3]
    vis = lm[:, 3:4]

    # normalize ด้วยจุดอ้างอิง 'สะโพกกลาง' และ scale ด้วยระยะไหล่ซ้าย-ขวา
    hip_center = _pair_midpoint(xyz[L_HIP], xyz[R_HIP])
    shoulder_dist = _dist(xyz[L_SHOULDER], xyz[R_SHOULDER])
    xyz_norm = (xyz - hip_center) / shoulder_dist

    angles = _joint_angles(xyz_norm)  # (12,)

    feat = np.concatenate([xyz_norm.reshape(-1), vis.reshape(-1), angles], axis=0)
    return feat
