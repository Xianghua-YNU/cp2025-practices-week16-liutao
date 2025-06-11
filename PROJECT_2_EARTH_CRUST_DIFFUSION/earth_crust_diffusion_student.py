import numpy as np  # 导入数值计算库
import matplotlib.pyplot as plt  # 导入绘图库

# ================================================
# 物理常数定义区域
# ================================================
D = 0.1      # 热扩散率 (m²/day) - 控制热量在介质中传播的速度
A = 10.0     # 年平均地表温度 (°C) - 地表温度的平均值
B = 12.0     # 地表温度振幅 (°C) - 地表温度变化的幅度
TAU = 365.0  # 年周期 (days) - 温度变化的周期长度
T_BOTTOM = 11.0  # 20米深处温度 (°C) - 底部边界固定温度
T_INITIAL = 10.0  # 初始温度 (°C) - 模拟开始时的初始温度分布
DEPTH_MAX = 20.0  # 最大深度 (m) - 模拟区域的深度范围

# ================================================
# 核心求解函数
# ================================================
def solve_earth_crust_diffusion(h=1.0, a=1.0, M=21, years=10):
    """
    求解地壳热扩散方程 (显式差分格式)
    
    参数:
        h (float): 空间步长 (m) [默认1.0m] - 深度方向的网格分辨率
        a (float): 时间步长比例因子 [默认1.0] - 用于计算时间步长
        M (int): 深度方向网格点数 [默认21] - 空间离散点数
        years (int): 总模拟年数 [默认10年] - 模拟的时间长度
    
    返回:
        tuple: (depth_array, temperature_matrix)
            - depth_array (ndarray): 深度数组 (m)
            - temperature_matrix (ndarray): 温度矩阵 [depth, time]
    """
    # -------------------------------------------------
    # 1. 参数计算和网格设置
    # -------------------------------------------------
    # 计算时间步长：dt = h * D / a²
    # 这个公式确保稳定性参数 r = D*dt/h² = (D * (h*D/a²)) / h² = D²/(a²h)
    dt = h * D / a**2
    
    # 计算稳定性参数 r = D*dt/h²
    r = D * dt / h**2
    print(f"空间步长 h = {h:.3f} m, 时间步长 dt = {dt:.4f} days, 稳定性参数 r = {r:.4f}")
    
    # 计算总模拟天数
    total_days = years * TAU
    
    # 计算时间步数 N = 总天数/时间步长 + 1
    # +1 是为了包括初始时间点
    N = int(total_days / dt) + 1
    
    # 创建深度数组：从0到20米，共M个点
    depth = np.arange(0, DEPTH_MAX + h, h)  # 使用arange确保步长为h
    
    # 创建时间数组：从0到总天数，共N个点
    time = np.linspace(0, total_days, N)
    
    # 初始化温度矩阵 (M×N)：深度方向×时间方向
    T = np.zeros((M, N))
    
    # -------------------------------------------------
    # 2. 初始条件和边界条件设置
    # -------------------------------------------------
    # 设置初始条件：所有深度点初始温度为10°C
    T[:, 0] = T_INITIAL
    
    # 设置上边界条件（地表温度）：随时间变化的正弦函数
    # T(0,t) = A + B*sin(2πt/τ)
    T[0, :] = A + B * np.sin(2 * np.pi * time / TAU)
    
    # 设置下边界条件（20米深度）：固定温度11°C
    T[-1, :] = T_BOTTOM
    
    # -------------------------------------------------
    # 3. 显式差分求解
    # -------------------------------------------------
    # 时间步进循环：从j=0到N-2（因为每个时间步计算下一时刻的值）
    for j in range(0, N-1):
        # 空间点循环：更新所有内部点（i=1到M-2）
        for i in range(1, M-1):
            # 显式差分公式：
            # T_i^{j+1} = T_i^j + r * (T_{i+1}^j - 2T_i^j + T_{i-1}^j)
            # 其中 r = D*dt/h²
            T[i, j+1] = T[i, j] + r * (T[i+1, j] + T[i-1, j] - 2*T[i, j])
    
    return depth, T

# ================================================
# 结果可视化函数
# ================================================
def plot_seasonal_profiles(depth, temperature, seasons=[90, 180, 270, 365]):
    """
    绘制季节性温度轮廓
    
    参数:
        depth (ndarray): 深度数组
        temperature (ndarray): 温度矩阵 [depth, time]
        seasons (list): 季节时间点 (一年中的天数) [默认:90(春),180(夏),270(秋),365(冬)]
    """
    # 创建图形
    plt.figure(figsize=(10, 8))
    
    # 遍历每个季节时间点
    for day in seasons:
        # 计算该时间点在时间数组中的索引
        # 使用整数除法确保索引有效
        j = day  
        
        # 确保索引不超出范围
        if j >= temperature.shape[1]:
            j = temperature.shape[1] - 1
        
        # 绘制该时间点的温度-深度曲线
        plt.plot(depth, temperature[:, j], 
                label=f'Day {day}', linewidth=2)
    
    # 添加图形标签和标题
    plt.xlabel('Depth (m)')  # x轴：深度
    plt.ylabel('Temperature (°C)')  # y轴：温度
    plt.title('Seasonal Temperature Profiles')  # 标题
    plt.grid(True)  # 显示网格
    plt.legend()  # 显示图例
    
    # 显示图形
    plt.show()

# ================================================
# 主程序
# ================================================
if __name__ == "__main__":
    # 1. 运行热扩散模拟
    depth, T = solve_earth_crust_diffusion(
        h=1.0,      # 空间步长1米
        a=1.0,      # 时间步长比例因子
        M=21,       # 深度方向21个点 (0,1,2,...,20)
        years=10     # 模拟10年
    )
    
    # 2. 绘制季节性温度轮廓
    plot_seasonal_profiles(depth, T)
