import numpy as np
import matplotlib.pyplot as plt

# 物理常数
D = 0.1      # 热扩散率 (m²/day)
A = 10.0     # 年平均地表温度 (°C)
B = 12.0     # 地表温度振幅 (°C)
TAU = 365.0  # 年周期 (days)
T_BOTTOM = 11.0  # 20米深处温度 (°C)
T_INITIAL = 10.0  # 初始温度 (°C)
DEPTH_MAX = 20.0  # 最大深度 (m)

def solve_earth_crust_diffusion(h=0.1, a=1.0, M=201, years=10):
    """
    求解地壳热扩散方程 (显式差分格式)
    
    参数:
        h (float): 空间步长 (m) [默认0.1m]
        a (float): 时间步长比例因子 [默认1.0]
        M (int): 深度方向网格点数 [默认201]
        years (int): 总模拟年数 [默认10年]
    
    返回:
        tuple: (depth_array, temperature_matrix)
            - depth_array (ndarray): 深度数组 (m)
            - temperature_matrix (ndarray): 温度矩阵 [depth, time]
    """
    # 计算时间步长和稳定性参数
    dt = h * D / a**2
    r = D * dt / h**2
    print(f"空间步长 h = {h:.3f} m, 时间步长 dt = {dt:.4f} days, 稳定性参数 r = {r:.4f}")
    
    # 计算时间网格参数
    total_days = years * TAU
    N = int(total_days / dt) + 1  # 时间步数
    
    # 创建网格
    depth = np.linspace(0, DEPTH_MAX, M)  # 深度坐标
    time = np.linspace(0, total_days, N)  # 时间坐标
    
    # 初始化温度矩阵 (深度×时间)
    T = np.zeros((M, N))
    
    # 设置初始条件
    T[:, 0] = T_INITIAL  # 初始温度分布
    T[0, :] = A + B * np.sin(2 * np.pi * time / TAU)  # 上边界条件
    T[-1, :] = T_BOTTOM  # 下边界条件
    
    # 显式差分格式求解
    for j in range(0, N-1):
        # 更新内部点 (从第1个到倒数第2个深度点)
        for i in range(1, M-1):
            T[i, j+1] = T[i, j] + r * (T[i+1, j] + T[i-1, j] - 2*T[i, j])
    
    return depth, T

def plot_seasonal_profiles(depth, T, seasons=[90, 180, 270, 360]):
    """
    绘制季节性温度轮廓
    
    参数:
        depth (ndarray): 深度数组
        T (ndarray): 温度矩阵 [depth, time]
        seasons (list): 季节时间点 (一年中的天数)
    """
    # 创建图形
    plt.figure(figsize=(10, 8))
    
    # 绘制各季节的温度轮廓
    for day in seasons:
        # 选择第10年对应的索引
        j = int(day * T.shape[1] / (365 * 10))
        plt.plot(depth, T[:, j], label=f'Day {day}', linewidth=2)
    
    # 添加标签和标题
    plt.xlabel('Depth (m)', fontsize=12)
    plt.ylabel('Temperature (°C)', fontsize=12)
    plt.title('Seasonal Temperature Profiles', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('seasonal_temperature_profiles.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    # 运行模拟
    depth, T = solve_earth_crust_diffusion()
    print(f"计算完成，温度场形状: depth={len(depth)}, time={T.shape[1]}")
    
    # 绘制季节性温度轮廓
    plot_seasonal_profiles(depth, T)
