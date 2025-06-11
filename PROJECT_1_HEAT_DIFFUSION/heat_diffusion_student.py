"""
学生模板：铝棒热传导问题
文件：heat_diffusion_student.py
重要：函数名称必须与参考答案一致！
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 物理参数
K = 237       # 热导率 (W/m/K)
C = 900       # 比热容 (J/kg/K)
rho = 2700    # 密度 (kg/m^3)
D = K/(C*rho)  # 热扩散系数
L = 1         # 铝棒长度 (m)
dx = 0.01     # 空间步长 (m)
dt = 0.5      # 时间步长 (s)
Nx = int(L/dx) + 1  # 空间格点数
Nt = 2000     # 时间步数
T0 = 100      # 初始温度


def basic_heat_diffusion():
    """
    任务1: 基本热传导模拟

    返回:
        np.ndarray: 温度分布数组
    """
    u = np.full(Nx, T0)
    u[0] = 0
    u[-1] = 0
    U = np.zeros((Nx, Nt))
    U[:, 0] = u
    r = D * dt / dx**2
    for j in range(Nt - 1):
        u = explicit_difference_method(u, D, dx, dt)
        U[:, j + 1] = u
    return U


def explicit_difference_method(u, D, dx, dt):
    Nx = len(u)
    r = D * dt / dx**2
    u_new = u.copy()
    for i in range(1, Nx - 1):
        u_new[i] = u[i] + r * (u[i + 1] - 2 * u[i] + u[i - 1])
    return u_new


def analytical_solution(n_terms=100):
    """
    任务2: 解析解函数

    参数:
        n_terms (int): 傅里叶级数项数

    返回:
        np.ndarray: 解析解温度分布
    """
    x = np.linspace(0, L, Nx)
    t = np.linspace(0, Nt * dt, Nt)
    X, T = np.meshgrid(x, t, indexing='ij')
    u = np.zeros((Nx, Nt))
    for n in range(1, 2 * n_terms, 2):
        k_n = n * np.pi / L
        u += (4 * T0 / (n * np.pi)) * np.sin(k_n * X) * np.exp(-k_n**2 * D * T)
    return u


def stability_analysis():
    """
    任务3: 数值解稳定性分析
    """
    # 稳定情况
    r_stable = 0.4
    dt_stable = r_stable * dx**2 / D
    u_stable = np.full(Nx, T0)
    u_stable[0] = 0
    u_stable[-1] = 0
    U_stable = np.zeros((Nx, Nt))
    U_stable[:, 0] = u_stable
    for j in range(Nt - 1):
        u_stable = explicit_difference_method(u_stable, D, dx, dt_stable)
        U_stable[:, j + 1] = u_stable

    # 不稳定情况
    r_unstable = 0.6
    dt_unstable = r_unstable * dx**2 / D
    u_unstable = np.full(Nx, T0)
    u_unstable[0] = 0
    u_unstable[-1] = 0
    U_unstable = np.zeros((Nx, Nt))
    U_unstable[:, 0] = u_unstable
    for j in range(Nt - 1):
        u_unstable = explicit_difference_method(u_unstable, D, dx, dt_unstable)
        U_unstable[:, j + 1] = u_unstable

    plot_3d_solution(U_stable, dx, dt_stable, Nt, "Stable Case (r = 0.4)")
    plot_3d_solution(U_unstable, dx, dt_unstable,
                     Nt, "Unstable Case (r = 0.6)")


def different_initial_condition():
    """
    任务4: 不同初始条件模拟

    返回:
        np.ndarray: 温度分布数组
    """
    u = np.zeros(Nx)
    u[:Nx//2] = T0
    u[0] = 0
    u[-1] = 0
    U = np.zeros((Nx, Nt))
    U[:, 0] = u
    for j in range(Nt - 1):
        u = explicit_difference_method(u, D, dx, dt)
        U[:, j + 1] = u
    return U


def heat_diffusion_with_cooling():
    """
    任务5: 包含牛顿冷却定律的热传导
    """
    h = 10  # 传热系数
    T_ambient = 0  # 环境温度
    u = np.full(Nx, T0)
    u[0] = 0
    u[-1] = 0
    U = np.zeros((Nx, Nt))
    U[:, 0] = u
    r = D * dt / dx**2
    for j in range(Nt - 1):
        u_new = u.copy()
        for i in range(1, Nx - 1):
            u_new[i] = u[i] + r * (u[i + 1] - 2 * u[i] + u[i - 1]) - \
                h * dt / (rho * C) * (u[i] - T_ambient)
        u = u_new
        U[:, j + 1] = u
    plot_3d_solution(U, dx, dt, Nt, "Heat Diffusion with Cooling")


def plot_3d_solution(u, dx, dt, Nt, title):
    """
    绘制3D温度分布图

    参数:
        u (np.ndarray): 温度分布数组
        dx (float): 空间步长
        dt (float): 时间步长
        Nt (int): 时间步数
        title (str): 图表标题

    返回:
        None

    示例:
        >>> u = np.zeros((100, 200))
        >>> plot_3d_solution(u, 0.01, 0.5, 200, "示例")
    """
    x = np.linspace(0, (u.shape[0] - 1) * dx, u.shape[0])
    t = np.linspace(0, Nt * dt, u.shape[1])
    X, T = np.meshgrid(x, t, indexing='ij')

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, T, u, cmap='viridis')
    ax.set_xlabel('Position (m)')
    ax.set_ylabel('Time (s)')
    ax.set_zlabel('Temperature (K)')
    ax.set_title(title)
    plt.show()


if __name__ == "__main__":
    """
    主函数 - 演示和测试各任务功能

    执行顺序:
    1. 基本热传导模拟
    2. 解析解计算
    3. 数值解稳定性分析
    4. 不同初始条件模拟
    5. 包含冷却效应的热传导

    注意:
        学生需要先实现各任务函数才能正常运行
    """
    print("=== 铝棒热传导问题学生实现 ===")
    # 1. 基本热传导模拟
    u_basic = basic_heat_diffusion()
    plot_3d_solution(u_basic, dx, dt, Nt, "Basic Heat Diffusion")

    # 2. 解析解计算
    u_analytical = analytical_solution()
    plot_3d_solution(u_analytical, dx, dt, Nt, "Analytical Solution")

    # 3. 数值解稳定性分析
    stability_analysis()

    # 4. 不同初始条件模拟
    u_different = different_initial_condition()
    plot_3d_solution(u_different, dx, dt, Nt, "Different Initial Condition")

    # 5. 包含冷却效应的热传导
    heat_diffusion_with_cooling()
