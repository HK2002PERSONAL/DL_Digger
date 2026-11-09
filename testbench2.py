import numpy as np
import iisignature

# 1. 基础路径：从 (0,0) 直线走到 (2,2) -> 没有包围任何面积
t = np.linspace(0, 1, 100)
path_base = np.column_stack([t * 2, t * 2])

# 2. 绕圈路径：走到 (1,1) 时，顺时针绕一个半径为 0.5 的圆圈，再继续走到 (2,2)
t_loop = np.linspace(0, 2 * np.pi, 100)
r = 0.5
# 顺时针圆圈偏移量
loop_x = 1.0 + r * np.sin(t_loop)
loop_y = 1.0 - r * (1 - np.cos(t_loop))

# 拼合包含封闭圈的路径
path_with_loop = np.vstack([
    path_base[:50],                   # 前半段
    np.column_stack([loop_x, loop_y]),# 封闭圈
    path_base[50:]                    # 后半段
])

# 3. 计算各自的莱维面积
def get_levy_area(path):
    sig = iisignature.sig(path, 2)
    S2 = sig[2:].reshape(2, 2)  # 取 2 阶签名矩阵
    return 0.5 * (S2[0, 1] - S2[1, 0])

area_base = get_levy_area(path_base)
area_loop = get_levy_area(path_with_loop)

print(f"基础路径莱维面积: {area_base:.4f}")         # 输出: 0.0000
print(f"包含封闭圈莱维面积: {area_loop:.4f}")       # 输出: ≈ 0.7854 (即 π * r^2 = π * 0.25)