#!/usr/bin/env python3
"""
学生模板：热传导方程数值解法比较
文件：heat_equation_methods_student.py
重要：函数名称必须与参考答案一致！
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import laplace
from scipy.integrate import solve_ivp
import scipy.linalg
import time

class HeatEquationSolver:
    """
    热传导方程求解器，实现四种不同的数值方法。
    
    求解一维热传导方程：du/dt = alpha * d²u/dx²
    边界条件：u(0,t) = 0, u(L,t) = 0
    初始条件：u(x,0) = phi(x)
    """
    
    def __init__(self, L=20.0, alpha=10.0, nx=21, T_final=25.0):
        """
        初始化热传导方程求解器。
        
        参数:
            L (float): 空间域长度 [0, L]
            alpha (float): 热扩散系数
            nx (int): 空间网格点数
            T_final (float): 最终模拟时间
        """
        self.L = L
        self.alpha = alpha
        self.nx = nx
        self.T_final = T_final
        
        # 空间网格
        self.x = np.linspace(0, L, nx)
        self.dx = L / (nx - 1)
        
        # 初始化解数组
        self.u_initial = self._set_initial_condition()
        
    def _set_initial_condition(self):
        """
        设置初始条件：u(x,0) = 1 当 10 <= x <= 11，否则为 0。
        
        返回:
            np.ndarray: 初始温度分布
        """
        u0 = np.zeros(self.nx)
        # 设置初始条件（10 <= x <= 11 区域为1）
        u0[(self.x >= 10) & (self.x <= 11)] = 1.0
        # 应用边界条件（虽然初始条件已经满足，但这里明确设置）
        u0[0] = 0.0
        u0[-1] = 0.0
        return u0
    
    def solve_explicit(self, dt=0.01, plot_times=None):
        """
        使用显式有限差分法（FTCS）求解。
        
        实现步骤:
        1. 检查稳定性条件
        2. 初始化解数组和时间
        3. 时间步进循环
        4. 使用laplace算子计算空间导数
        5. 更新解并应用边界条件
        6. 存储指定时间点的解
        """
        if plot_times is None:
            plot_times = [0, 1, 5, 15, 25]
            
        # 计算稳定性参数
        r = self.alpha * dt / (self.dx ** 2)
        
        # 检查稳定性条件
        if r > 0.5:
            print(f"警告：显式方法稳定性条件不满足 (r={r:.4f} > 0.5)，结果可能不稳定")
            print(f"建议减小时间步长至: {0.5 * self.dx**2 / self.alpha:.6f}")
        
        # 初始化解数组
        u = self.u_initial.copy()
        t = 0.0
        nt = int(self.T_final / dt) + 1  # 总时间步数
        
        # 创建结果存储字典
        results = {
            'method': 'Explicit FTCS',
            'times': [],
            'solutions': [],
            'dt': dt,
            'r': r,
            'execution_time': 0.0
        }
        
        # 存储初始条件
        if 0 in plot_times:
            results['times'].append(0.0)
            results['solutions'].append(u.copy())
        
        start_time = time.time()
        
        # 时间步进循环
        for n in range(1, nt):
            # 使用laplace计算空间二阶导数
            laplace_u = laplace(u, mode='constant', cval=0.0)
            
            # 更新解
            u += r * laplace_u
            
            # 应用边界条件
            u[0] = 0.0
            u[-1] = 0.0
            
            t = n * dt
            
            # 检查是否到达绘图时间点
            for plot_time in plot_times:
                # 使用时间容差判断
                if abs(t - plot_time) < dt/2 and plot_time not in results['times']:
                    results['times'].append(t)
                    results['solutions'].append(u.copy())
        
        # 确保最终时间被记录
        if self.T_final not in results['times']:
            results['times'].append(self.T_final)
            results['solutions'].append(u.copy())
            
        end_time = time.time()
        results['execution_time'] = end_time - start_time
        
        return results
    
    def solve_implicit(self, dt=0.1, plot_times=None):
        """
        使用隐式有限差分法（BTCS）求解。
        
        实现步骤:
        1. 计算扩散数 r
        2. 构建三对角系数矩阵
        3. 时间步进循环
        4. 构建右端项
        5. 求解线性系统
        6. 更新解并应用边界条件
        """
        if plot_times is None:
            plot_times = [0, 1, 5, 15, 25]
            
        # 计算扩散数 r
        r = self.alpha * dt / (self.dx ** 2)
        
        # 构建三对角矩阵（内部节点）
        n = self.nx - 2  # 内部节点数
        nt = int(self.T_final / dt) + 1  # 总时间步数
        
        # 主对角线
        main_diag = np.ones(n) * (1 + 2 * r)
        # 下对角线
        lower_diag = np.ones(n - 1) * (-r)
        # 上对角线
        upper_diag = np.ones(n - 1) * (-r)
        
        # 将三对角矩阵组合为带状矩阵格式
        A_banded = np.zeros((3, n))
        A_banded[0, 1:] = upper_diag
        A_banded[1, :] = main_diag
        A_banded[2, :-1] = lower_diag
        
        # 初始化解数组
        u = self.u_initial.copy()
        
        # 创建结果存储字典
        results = {
            'method': 'Implicit BTCS',
            'times': [],
            'solutions': [],
            'dt': dt,
            'r': r,
            'execution_time': 0.0
        }
        
        # 存储初始条件
        if 0 in plot_times:
            results['times'].append(0.0)
            results['solutions'].append(u.copy())
        
        start_time = time.time()
        
        # 时间步进循环
        for n in range(1, nt):
            # 构建右端项（内部节点）
            rhs = u[1:-1].copy()
            
            # 求解线性系统
            u_internal = scipy.linalg.solve_banded((1, 1), A_banded, rhs)
            
            # 更新解
            u[1:-1] = u_internal
            # 应用边界条件
            u[0] = 0.0
            u[-1] = 0.0
            
            t = n * dt
            
            # 检查是否到达绘图时间点
            for plot_time in plot_times:
                # 使用时间容差判断
                if abs(t - plot_time) < dt/2 and plot_time not in results['times']:
                    results['times'].append(t)
                    results['solutions'].append(u.copy())
        
        # 确保最终时间被记录
        if self.T_final not in results['times']:
            results['times'].append(self.T_final)
            results['solutions'].append(u.copy())
            
        end_time = time.time()
        results['execution_time'] = end_time - start_time
        
        return results
    
    def solve_crank_nicolson(self, dt=0.5, plot_times=None):
        """
        使用Crank-Nicolson方法求解。
        
        实现步骤:
        1. 计算扩散数 r
        2. 构建左端矩阵 A
        3. 时间步进循环
        4. 构建右端向量
        5. 求解线性系统 A * u^{n+1} = rhs
        6. 更新解并应用边界条件
        """
        if plot_times is None:
            plot_times = [0, 1, 5, 15, 25]
            
        # 计算扩散数 r
        r = self.alpha * dt / (self.dx ** 2)
        nt = int(self.T_final / dt) + 1  # 总时间步数
        
        # 构建左端矩阵 A（内部节点）
        n = self.nx - 2  # 内部节点数
        
        # 主对角线
        main_diag = np.ones(n) * (1 + r)
        # 下对角线
        lower_diag = np.ones(n - 1) * (-r/2)
        # 上对角线
        upper_diag = np.ones(n - 1) * (-r/2)
        
        # 将三对角矩阵组合为带状矩阵格式
        A_banded = np.zeros((3, n))
        A_banded[0, 1:] = upper_diag
        A_banded[1, :] = main_diag
        A_banded[2, :-1] = lower_diag
        
        # 初始化解数组
        u = self.u_initial.copy()
        
        # 创建结果存储字典
        results = {
            'method': 'Crank-Nicolson',
            'times': [],
            'solutions': [],
            'dt': dt,
            'r': r,
            'execution_time': 0.0
        }
        
        # 存储初始条件
        if 0 in plot_times:
            results['times'].append(0.0)
            results['solutions'].append(u.copy())
        
        start_time = time.time()
        
        # 时间步进循环
        for n in range(1, nt):
            # 构建右端向量
            u_internal = u[1:-1]
            # 正确的右端项公式
            rhs = (r/2) * u[:-2] + (1 - r) * u_internal + (r/2) * u[2:]
            
            # 求解线性系统
            u_internal_new = scipy.linalg.solve_banded((1, 1), A_banded, rhs)
            
            # 更新解
            u[1:-1] = u_internal_new
            # 应用边界条件
            u[0] = 0.0
            u[-1] = 0.0
            
            t = n * dt
            
            # 检查是否到达绘图时间点
            for plot_time in plot_times:
                # 使用时间容差判断
                if abs(t - plot_time) < dt/2 and plot_time not in results['times']:
                    results['times'].append(t)
                    results['solutions'].append(u.copy())
        
        # 确保最终时间被记录
        if self.T_final not in results['times']:
            results['times'].append(self.T_final)
            results['solutions'].append(u.copy())
            
        end_time = time.time()
        results['execution_time'] = end_time - start_time
        
        return results
    
    def _heat_equation_ode(self, t, u_internal):
        """
        用于solve_ivp方法的ODE系统。
        
        实现步骤:
        1. 重构包含边界条件的完整解
        2. 使用laplace计算二阶导数
        3. 返回内部节点的导数
        """
        # 重构完整解向量（包含边界条件）
        u_full = np.zeros(self.nx)
        u_full[1:-1] = u_internal
        u_full[0] = 0.0  # 左边界
        u_full[-1] = 0.0  # 右边界
        
        # 使用 laplace 计算二阶导数
        laplace_u = laplace(u_full, mode='constant', cval=0.0)
        d2u_dx2 = laplace_u / (self.dx ** 2)
        
        # 返回内部节点的时间导数：alpha * d²u/dx²
        return self.alpha * d2u_dx2[1:-1]
    
    def solve_with_solve_ivp(self, method='BDF', plot_times=None):
        """
        使用scipy.integrate.solve_ivp求解。
        
        实现步骤:
        1. 提取内部节点初始条件
        2. 调用solve_ivp求解ODE系统
        3. 重构包含边界条件的完整解
        4. 返回结果
        """
        if plot_times is None:
            plot_times = [0, 1, 5, 15, 25]
            
        # 提取内部节点初始条件
        u0_internal = self.u_initial[1:-1].copy()
        
        # 添加初始时间
        if 0 not in plot_times:
            plot_times = [0] + list(plot_times)
        
        # 调用 solve_ivp 求解
        start_time = time.time()
        sol = solve_ivp(
            fun=self._heat_equation_ode,
            t_span=(0, self.T_final),
            y0=u0_internal,
            method=method,
            t_eval=plot_times,
            rtol=1e-6,
            atol=1e-8
        )
        end_time = time.time()
        
        # 重构包含边界条件的完整解
        solutions = []
        for i in range(sol.y.shape[1]):
            u_full = np.zeros(self.nx)
            u_full[1:-1] = sol.y[:, i]
            solutions.append(u_full)
        
        # 创建结果存储字典
        results = {
            'method': f'solve_ivp ({method})',
            'times': sol.t.tolist(),
            'solutions': solutions,
            'dt': 'adaptive',
            'r': 'N/A',
            'execution_time': end_time - start_time
        }
        
        return results
    
    def compare_methods(self, dt_explicit=0.01, dt_implicit=0.1, dt_cn=0.5, 
                       ivp_method='BDF', plot_times=None):
        """
        比较所有四种数值方法。
        
        实现步骤:
        1. 调用所有四种求解方法
        2. 记录计算时间和稳定性参数
        3. 返回比较结果
        """
        if plot_times is None:
            plot_times = [0, 1, 5, 15, 25]
            
        # 打印求解信息
        print(f"\n{'='*50}")
        print(f"热传导方程求解器比较 (L={self.L}, α={self.alpha}, nx={self.nx}, T_final={self.T_final})")
        print(f"时间步长设置: 显式={dt_explicit}, 隐式={dt_implicit}, C-N={dt_cn}, solve_ivp={ivp_method}")
        print(f"绘图时间点: {plot_times}")
        print('='*50)
        
        # 调用四种求解方法
        results = {}
        
        print("\n求解显式方法...")
        results['explicit'] = self.solve_explicit(dt=dt_explicit, plot_times=plot_times)
        
        print("求解隐式方法...")
        results['implicit'] = self.solve_implicit(dt=dt_implicit, plot_times=plot_times)
        
        print("求解Crank-Nicolson方法...")
        results['crank_nicolson'] = self.solve_crank_nicolson(dt=dt_cn, plot_times=plot_times)
        
        print(f"求解solve_ivp方法 ({ivp_method})...")
        results['solve_ivp'] = self.solve_with_solve_ivp(method=ivp_method, plot_times=plot_times)
        
        # 打印每种方法的计算时间和稳定性参数
        print("\n方法比较结果:")
        print("-"*70)
        print(f"{'方法':<25} | {'时间步长':<10} | {'r值':<10} | {'执行时间(s)':<12} | {'状态'}")
        print("-"*70)
        
        return results
    
    def plot_comparison(self, methods_results, save_figure=False, filename='heat_equation_comparison.png'):
        """
        绘制所有方法的比较图。
        
        实现步骤:
        1. 创建2x2子图
        2. 为每种方法绘制不同时间的解
        3. 设置图例、标签和标题
        4. 可选保存图像
        """
        # 获取所有方法的时间点（取第一个方法的时间点作为参考）
        times = methods_results['explicit']['times']
        
        # 创建 2x2 子图
        fig, axs = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'热传导方程数值解法比较 (L={self.L}, α={self.alpha}, nx={self.nx})', fontsize=16)
        
        # 为每种方法绘制解曲线
        methods = ['explicit', 'implicit', 'crank_nicolson', 'solve_ivp']
        titles = ['显式方法 (FTCS)', '隐式方法 (BTCS)', 'Crank-Nicolson 方法', 'solve_ivp 方法']
        
        # 颜色和线型
        colors = ['blue', 'green', 'red', 'purple', 'orange']
        linestyles = ['-', '--', '-.', ':']
        
        for i, method_key in enumerate(methods):
            ax = axs[i//2, i%2]
            res = methods_results[method_key]
            
            # 对于每个时间点
            for j, t in enumerate(times):
                # 找到最接近的时间索引
                idx = np.argmin(np.abs(np.array(res['times']) - t))
                closest_time = res['times'][idx]
                solution = res['solutions'][idx]
                
                # 绘制曲线
                color_idx = j % len(colors)
                linestyle_idx = j // len(colors) % len(linestyles)
                
                ax.plot(self.x, solution, 
                        color=colors[color_idx], 
                        linestyle=linestyles[linestyle_idx],
                        label=f't={closest_time:.1f}s')
            
            ax.set_title(f"{res['method']}\n(计算时间: {res['execution_time']:.4f}s")
            ax.set_xlabel('位置 x')
            ax.set_ylabel('温度 u(x,t)')
            ax.grid(True, alpha=0.3)
            ax.legend()
            ax.set_xlim(0, self.L)
            ax.set_ylim(-0.1, 1.1)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        if save_figure:
            plt.savefig(filename, dpi=300)
            print(f"图像已保存为: {filename}")
        
        plt.show()
    
    def analyze_accuracy(self, methods_results, reference_method='solve_ivp'):
        """
        分析不同方法的精度。
        
        实现步骤:
        1. 选择参考解
        2. 计算其他方法与参考解的误差
        3. 统计最大误差和平均误差
        4. 返回分析结果
        """
        # 验证参考方法存在
        if reference_method not in methods_results:
            print(f"错误: 参考方法 '{reference_method}' 不存在")
            return None
        
        # 获取参考解
        ref_results = methods_results[reference_method]
        ref_times = ref_results['times']
        ref_solutions = ref_results['solutions']
        
        # 创建精度分析字典
        accuracy_results = {
            'reference_method': ref_results['method'],
            'comparisons': {}
        }
        
        print(f"\n精度分析 (参考方法: {ref_results['method']})")
        print("-"*70)
        print(f"{'方法':<25} | {'时间点':<10} | {'最大误差':<12} | {'平均误差':<12} | {'2-范数误差':<12}")
        print("-"*70)
        
        # 计算各方法与参考解的误差
        for key, res in methods_results.items():
            if key == reference_method:
                continue
                
            # 存储误差数据
            errors = {
                'max_errors': [],
                'mean_errors': [],
                'norm2_errors': [],
                'times': []
            }
            
            # 对于每个参考时间点
            for i, t in enumerate(ref_times):
                # 找到参考解
                ref_u = ref_solutions[i]
                
                # 在当前方法中找到最接近的时间点的解
                idx = np.argmin(np.abs(np.array(res['times']) - t))
                closest_time = res['times'][idx]
                u = res['solutions'][idx]
                
                # 计算误差
                error = np.abs(u - ref_u)
                max_error = np.max(error)
                mean_error = np.mean(error)
                norm2_error = np.linalg.norm(u - ref_u)
                
                errors['times'].append(closest_time)
                errors['max_errors'].append(max_error)
                errors['mean_errors'].append(mean_error)
                errors['norm2_errors'].append(norm2_error)
                
                # 打印当前时间点的误差
                print(f"{res['method']:<25} | {t:<10.1f} | {max_error:<12.6f} | {mean_error:<12.6f} | {norm2_error:<12.6f}")
            
            accuracy_results['comparisons'][res['method']] = errors
            print("-"*70)
        
        return accuracy_results


def main():
    """
    HeatEquationSolver类的演示。
    """
    # 创建求解器实例
    solver = HeatEquationSolver(L=20.0, alpha=10.0, nx=101, T_final=25.0)
    
    # 设置绘图时间点
    plot_times = [0, 0.5, 1, 2, 5, 10, 15, 20, 25]
    
    # 比较所有方法
    results = solver.compare_methods(
        dt_explicit=0.001,   # 显式方法需要小时间步长
        dt_implicit=0.1,     # 隐式方法可以使用较大时间步长
        dt_cn=0.5,           # C-N方法可以使用更大的时间步长
        ivp_method='Radau',   # 使用高精度ODE求解器
        plot_times=plot_times
    )
    
    # 绘制比较图
    solver.plot_comparison(results, save_figure=True)
    
    # 分析精度
    accuracy = solver.analyze_accuracy(results)
    
    return solver, results, accuracy


if __name__ == "__main__":
    solver, results, accuracy = main()
