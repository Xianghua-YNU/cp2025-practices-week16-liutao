import numpy as np
import matplotlib.pyplot as plt

def solve_earth_crust_diffusion():
    """
    实现显式差分法求解地壳热扩散问题
    
    返回:
        tuple: (depth_array, temperature_matrix)
        depth_array: 深度坐标数组 (m)
        temperature_matrix: 温度场矩阵 (°C)
    """
    # 物理参数
    D = 0.1          # 热扩散率 (m²/day)
    A = 10.0         # 年平均地表温度 (°C)
    B = 12.0         # 地表温度振幅 (°C)
    tau = 365.0      # 温度变化周期 (days)
    T_bottom = 11.0  # 20米深处固定温度 (°C)
    z_max = 20.0     # 最大深度 (m)
    
    # 网格参数
    dz = 0.1         # 空间步长 (m) - 满足稳定性条件
    dt = 0.1         # 时间步长 (days) - 满足稳定性条件
    n_z = int(z_max / dz) + 1  # 空间网格点数
    total_years = 10  # 模拟总年数
    total_days = total_years * tau
    n_t = int(total_days / dt) + 1  # 时间步数
    
    # 计算稳定性参数 r
    r = D * dt / (dz**2)
    print(f"稳定性参数 r = {r:.4f} (应 ≤ 0.5)")
    
    # 创建网格
    depth = np.linspace(0, z_max, n_z)  # 深度坐标
    time = np.linspace(0, total_days, n_t)  # 时间坐标
    
    # 初始化温度场
    T = np.zeros((n_t, n_z))
    
    # 设置初始条件
    T[0, :] = 10.0  # 初始温度分布
    T[0, 0] = A + B * np.sin(2 * np.pi * time[0] / tau)  # 上边界
    T[0, -1] = T_bottom  # 下边界
    
    # 显式差分格式求解
    for n in range(0, n_t-1):
        # 应用上边界条件
        T[n+1, 0] = A + B * np.sin(2 * np.pi * time[n+1] / tau)
        
        # 应用下边界条件
        T[n+1, -1] = T_bottom
        
        # 内部点更新
        for i in range(1, n_z-1):
            T[n+1, i] = T[n, i] + r * (T[n, i+1] - 2*T[n, i] + T[n, i-1])
    
    return depth, T

def plot_seasonal_profiles(depth, T):
    """
    绘制第10年四季的温度轮廓图
    
    参数:
        depth: 深度数组
        T: 温度矩阵
    """
    # 计算时间参数
    dt = 0.1  # 时间步长 (days)
    total_days = 10 * 365
    n_t = T.shape[0]
    
    # 选择第10年的四个时间点（代表四季）
    # 春季 (3月21日左右) - 第9年结束后的第80天
    spring_idx = int((9*365 + 80) / dt)
    # 夏季 (6月21日左右) - 第9年结束后的第172天
    summer_idx = int((9*365 + 172) / dt)
    # 秋季 (9月23日左右) - 第9年结束后的第265天
    fall_idx = int((9*365 + 265) / dt)
    # 冬季 (12月22日左右) - 第9年结束后的第355天
    winter_idx = int((9*365 + 355) / dt)
    
    # 确保索引在范围内
    spring_idx = min(spring_idx, n_t-1)
    summer_idx = min(summer_idx, n_t-1)
    fall_idx = min(fall_idx, n_t-1)
    winter_idx = min(winter_idx, n_t-1)
    
    # 创建图形
    plt.figure(figsize=(10, 7))
    
    # 绘制四季温度轮廓
    plt.plot(T[spring_idx, :], depth, 'g-', linewidth=2, label='Spring (Day 80)')
    plt.plot(T[summer_idx, :], depth, 'r-', linewidth=2, label='Summer (Day 172)')
    plt.plot(T[fall_idx, :], depth, 'b-', linewidth=2, label='Fall (Day 265)')
    plt.plot(T[winter_idx, :], depth, 'c-', linewidth=2, label='Winter (Day 355)')
    
    # 反转y轴使深度向下为正
    plt.gca().invert_yaxis()
    
    # 添加标签和标题
    plt.xlabel('Temperature (°C)', fontsize=12)
    plt.ylabel('Depth (m)', fontsize=12)
    plt.title('Seasonal Temperature Profiles at Year 10', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 设置坐标轴范围
    plt.xlim(-2, 22)
    plt.ylim(20, 0)
    
    plt.tight_layout()
    plt.savefig('seasonal_temperature_profiles.png', dpi=300)
    plt.show()

def analyze_amplitude_phase(depth, T):
    """
    分析温度振幅衰减和相位延迟随深度的变化
    
    参数:
        depth: 深度数组
        T: 温度矩阵
    """
    # 提取第10年的数据
    start_idx = int(9 * 365 / 0.1)  # 第10年开始的时间索引
    T_year10 = T[start_idx:, :]
    
    # 计算每个深度的振幅和相位
    amplitudes = np.zeros(len(depth))
    phases = np.zeros(len(depth))
    
    for i in range(len(depth)):
        # 获取该深度全年的温度变化
        temp_series = T_year10[:, i]
        
        # 计算振幅 (最大值与最小值的差的一半)
        amplitudes[i] = (np.max(temp_series) - np.min(temp_series)) / 2
        
        # 计算相位延迟 (找到最大值出现的时间)
        max_idx = np.argmax(temp_series)
        phases[i] = max_idx * 0.1  # 时间步长为0.1天
    
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
    plt.plot(phases - phases[0], depth, 'r-', linewidth=2)
    plt.gca().invert_yaxis()
    plt.xlabel('Phase Delay (days)', fontsize=12)
    plt.ylabel('Depth (m)', fontsize=12)
    plt.title('Phase Delay with Depth', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('amplitude_phase_analysis.png', dpi=300)
    plt.show()
    
    return amplitudes, phases

if __name__ == "__main__":
    # 运行模拟
    depth, T = solve_earth_crust_diffusion()
    print(f"计算完成，温度场形状: {T.shape}")
    
    # 绘制四季温度轮廓
    plot_seasonal_profiles(depth, T)
    
    # 分析振幅衰减和相位延迟
    amplitudes, phases = analyze_amplitude_phase(depth, T)
