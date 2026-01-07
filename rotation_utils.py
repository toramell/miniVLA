import torch
import torch.nn.functional as F
import numpy as np

def euler_to_rot6d(euler_angles):
    """
    オイラー角 (B, 3) -> 6D回転表現 (B, 6)
    Euler (roll, pitch, yaw) -> Rotation Matrix (3x3) -> 取捨選択して6Dへ
    """
    batch_size = euler_angles.shape[0]
    x, y, z = euler_angles[:, 0], euler_angles[:, 1], euler_angles[:, 2]

    # 回転行列の構築 (Z-Y-X rotation convention for LIBERO/Robosuite)
    cx, sx = torch.cos(x), torch.sin(x)
    cy, sy = torch.cos(y), torch.sin(y)
    cz, sz = torch.cos(z), torch.sin(z)

    # 3x3 回転行列
    # R = Rz(yaw) * Ry(pitch) * Rx(roll)
    m00 = cy * cz
    m01 = cz * sx * sy - cx * sz
    m02 = cx * cz * sy + sx * sz
    m10 = cy * sz
    m11 = cx * cz + sx * sy * sz
    m12 = -cz * sx + cx * sy * sz
    m20 = -sy
    m21 = cy * sx
    m22 = cx * cy

    # 最初の2列 (Column 0 and 1) を取り出す -> 6D表現
    # matrix: (B, 3, 3) -> vector: (B, 6)
    # col0: m00, m10, m20
    # col1: m01, m11, m21
    
    rot6d = torch.stack([m00, m10, m20, m01, m11, m21], dim=1)
    return rot6d

def rot6d_to_euler(rot6d):
    """
    6D回転表現 (B, 6) -> オイラー角 (B, 3)
    """
    # 1. 6Dベクトルから直交行列を復元 (Gram-Schmidt process)
    x_raw = rot6d[:, 0:3]  # 1列目
    y_raw = rot6d[:, 3:6]  # 2列目

    # xを正規化
    x = F.normalize(x_raw, dim=-1)
    # yからx成分を引いて直交化し、正規化
    z = torch.cross(x, y_raw, dim=-1)
    z = F.normalize(z, dim=-1)
    y = torch.cross(z, x, dim=-1)

    # 回転行列 (B, 3, 3)
    # [x, y, z]
    m00, m10, m20 = x[:, 0], x[:, 1], x[:, 2]
    m01, m11, m21 = y[:, 0], y[:, 1], y[:, 2]
    m02, m12, m22 = z[:, 0], z[:, 1], z[:, 2]

    # 2. 回転行列からオイラー角 (Roll-Pitch-Yaw / xyz) を復元
    # sy = sqrt(m00*m00 + m10*m10)
    sy = torch.sqrt(m00*m00 + m10*m10)

    # 特異点判定なしの簡易実装 (Robosuite互換)
    # atan2(y, x)
    roll = torch.atan2(m21, m22)
    pitch = torch.atan2(-m20, sy)
    yaw = torch.atan2(m10, m00)

    return torch.stack([roll, pitch, yaw], dim=1)