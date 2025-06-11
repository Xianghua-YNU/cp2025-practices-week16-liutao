import numpy as np
import matplotlib.pyplot as plt

# 物理常数
D = 0.1  # 热扩散率 (m²/day)
A = 10.0  # 年平均地表温度 (°C)
B = 12.0  # 地表温度振幅 (°C)
TAU = 365.0  # 年周期 (days)
T_BOTTOM = 11.0  # 20米深处温度 (°C)
T_INITIAL = 10.0  # 初始温度 (°C)
DEPTH_MAX = 20.0  # 最大深度 (m)

def solve_earth_crust_diffusion(h=0.1, r=0.1, years=10):
    """
    求解地壳热扩散方程 (显式差分格式)
    
    参数:
        h (float): 空间步长 (m) [默认0.1m]
        r (float): 稳定性参数 r = D * dt / h² [默认0.1]
        years (int): 总模拟年数 [默认10年]
    
    返回:
        tuple: (depth_array, temperature_matrix)
            - depth_array (ndarray): 深度数组 (m)
            - temperature_matrix (ndarray): 温度矩阵 [depth, time]
    """
    # 计算时间步长
    dt = r * h**2 / D
    print(f"空间步长 h = {h:.3f} m, 时间步长 dt = {dt:.4f} days, 稳定性参数 r = {r:.3f}")
    
    # 计算空间和时间网格参数
    M = int(DEPTH_MAX / h) + 1  # 深度方向网格点数
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

def plot_seasonal_profiles(depth, T, seasons=[80, 172, 265, 355], year=9):
    """
    绘制季节性温度轮廓
    
    参数:
        depth (ndarray): 深度数组
        T (ndarray): 温度矩阵 [depth, time]
        seasons (list): 季节时间点 (一年中的天数)
        year (int): 选择的年份 (0-based, 0=第一年)
    """
    # 计算时间步长
    dt = (years * TAU) / (T.shape[1] - 1)
    
    # 选择指定年份的季节时间点
    seasonal_profiles = []
    for day in seasons:
        # 计算该时间点的索引
        time_idx = int((year * TAU + day) / dt)
        if time_idx >= T.shape[1]:
            time_idx = T.shape[1] - 1
        seasonal_profiles.append(T[:, time_idx])
    
    # 创建图形
    plt.figure(figsize=(10, 8))
    
    # 绘制各季节的温度轮廓
    labels = ['Spring (Day 80)', 'Summer (Day 172)', 'Fall (Day 265)', 'Winter (Day 355)']
    colors = ['g-', 'r-', 'b-', 'c-']
    
    for i, profile in enumerate(seasonal_profiles):
        plt.plot(profile, depth, colors[i], linewidth=2, label=labels[i])
    
    # 反转y轴使深度向下为正
    plt.gca().invert_yaxis()
    
    # 添加标签和标题
    plt.xlabel('Temperature (°C)', fontsize=12)
    plt.ylabel('Depth (m)', fontsize=12)
    plt.title(f'Seasonal Temperature Profiles at Year {year+1}', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 设置坐标轴范围
    plt.xlim(-3, 23)
    plt.ylim(20, 0)
    
    plt.tight_layout()
    plt.savefig('seasonal_temperature_profiles.png', dpi=300)
    plt.show()

def analyze_amplitude_phase(depth, T, year=9):
    """
    分析温度振幅衰减和相位延迟随深度的变化
    
    参数:
        depth (ndarray): 深度数组
        T (ndarray): 温度矩阵 [depth, time]
        year (int): 分析的年份 (0-based)
    """
    # 计算时间步长
    dt = (years * TAU) / (T.shape[1] - 1)
    
    # 提取指定年份的数据
    start_idx = int(year * TAU / dt)
    end_idx = int((year + 1) * TAU / dt)
    T_year = T[:, start_idx:end_idx]
    
    # 计算每个深度的振幅和相位
    amplitudes = np.zeros(len(depth))
    phases = np.zeros(len(depth))
    
    for i in range(len(depth)):
        # 获取该深度全年的温度变化
        temp_series = T_year[i, :]
        
        # 计算振幅 (最大值与最小值的差的一半)
        amplitudes[i] = (np.max(temp_series) - np.min(temp_series)) / 2
        
        # 计算相位延迟 (找到最大值出现的时间)
        max_idx = np.argmax(temp_series)
        phases[i] = max_idx * dt  # 转换为天数
    
    # 计算相位延迟（相对于地表）
    phase_delay = phases - phases[0]
    
    # 绘制振幅衰减
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(amplitudes, depth, 'b-', linewidth=2)
    plt.gca().invert_yaxis()
    plt.xlabel('Temperature Amplitude (°C)', fontsize=12)
    plt.ylabel('Depth (m)', fontsize=12)
    plt.title('Amplitude Attenuation with Depth', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 绘制相位延迟
    plt.subplot(1, 2, 2)
    plt.plot(phase_delay, depth, 'r-', linewidth=2)
    plt.gca().invert_yaxis()
    plt.xlabel('Phase Delay (days)', fontsize=12)
    plt.ylabel('Depth (m)', fontsize=12)
    plt.title('Phase Delay with Depth', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('amplitude_phase_analysis.png', dpi=300)
    plt.show()
    
    return amplitudes, phase_delay

if __name__ == "__main__":
    # 运行模拟
    years = 10
    depth, T = solve_earth_crust_diffusion(h=0.1, r=0.1, years=years)
    print(f"计算完成，温度场形状: {T.shape}")
    
    # 绘制四季温度轮廓
    plot_seasonal_profiles(depth, T, year=9)  # 第10年
    
    # 分析振幅衰减和相位延迟
    amplitudes, phase_delay = analyze_amplitude_phase(depth, T, year=9)
    
    # 输出关键深度处的分析结果
    key_depths = [0, 2, 5, 10, 15, 20]
    print("\n深度 | 温度振幅 (°C) | 相位延迟 (天)")
    print("-----------------------------------")
    for d in key_depths:
        idx = np.argmin(np.abs(depth - d))
        print(f"{depth[idx]:.1f} | {amplitudes[idx]:.3f} | {phase_delay[idx]:.2f}")
