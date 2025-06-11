"""学生模板：量子隧穿效应
文件：quantum_tunneling_student.py
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

# 设置同时支持中文和数学符号
plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文显示
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题
plt.rcParams['mathtext.fontset'] = 'stix'     # 数学字体设置

class QuantumTunnelingSolver:
    def __init__(self, Nx=220, Nt=300, x0=40, k0=0.5, d=10, barrier_width=3, barrier_height=1.0):
        self.Nx = Nx
        self.Nt = Nt
        self.x0 = x0
        self.k0 = k0
        self.d = d
        self.barrier_width = barrier_width
        self.barrier_height = barrier_height
        self.x = np.arange(self.Nx)
        self.V = self.setup_potential()
        self.C = np.zeros((self.Nx, self.Nt), complex)
        self.B = np.zeros((self.Nx, self.Nt), complex)

    def wavefun(self, x):
        """高斯波包函数"""
        return np.exp(self.k0*1j*x)*np.exp(-(x-self.x0)**2*np.log10(2)/self.d**2)

    def setup_potential(self):
        """设置势垒"""
        self.V = np.zeros(self.Nx)
        self.V[self.Nx//2:self.Nx//2+self.barrier_width] = self.barrier_height
        return self.V

    def build_coefficient_matrix(self):
        """构建Crank-Nicolson方案的系数矩阵"""
        A = np.diag(-2+2j-self.V) + np.diag(np.ones(self.Nx-1),1) + np.diag(np.ones(self.Nx-1),-1)
        return A

    def solve_schrodinger(self):
        """使用Crank-Nicolson方法求解一维含时薛定谔方程"""
        A = self.build_coefficient_matrix()
        
        self.B[:,0] = self.wavefun(self.x)
        
        for t in range(self.Nt-1):
            self.C[:,t+1] = 4j*np.linalg.solve(A, self.B[:,t])
            self.B[:,t+1] = self.C[:,t+1] - self.B[:,t]
        
        return self.x, self.V, self.B, self.C

    def calculate_coefficients(self):
        """计算透射和反射系数"""
        barrier_position = len(self.x)//2
        transmitted_prob = np.sum(np.abs(self.B[barrier_position+self.barrier_width:, -1])**2)
        reflected_prob = np.sum(np.abs(self.B[:barrier_position, -1])**2)
        total_prob = np.sum(np.abs(self.B[:, -1])**2)
        return transmitted_prob/total_prob, reflected_prob/total_prob

    def plot_evolution(self, time_indices=None):
        """在特定时间点绘制波函数演化"""
        if time_indices is None:
            Nt = self.B.shape[1]
            time_indices = [0, Nt//4, Nt//2, 3*Nt//4, Nt-1]
        
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        axes = axes.flatten()
        
        # 添加包含势垒参数的总体标题
        fig.suptitle(f'量子隧穿演化 - 势垒宽度: {self.barrier_width}, 势垒高度: {self.barrier_height}', 
                     fontsize=14, fontweight='bold')
        
        for i, t_idx in enumerate(time_indices):
            if i < len(axes):
                ax = axes[i]
                
                # 绘制概率密度
                prob_density = np.abs(self.B[:, t_idx])**2
                ax.plot(self.x, prob_density, 'b-', linewidth=2, 
                       label=r'$|\psi|^2$ 当 t={}'.format(t_idx))
                
                # 绘制势能
                ax.plot(self.x, self.V, 'k-', linewidth=2, 
                       label=f'势垒 (宽度={self.barrier_width}, 高度={self.barrier_height})')
                
                ax.set_xlabel('位置')
                ax.set_ylabel('概率密度')
                ax.set_title(f'时间步长: {t_idx}')
                ax.legend(loc='upper right')
                ax.grid(True, alpha=0.3)
        
        # 移除未使用的子图
        for i in range(len(time_indices), len(axes)):
            fig.delaxes(axes[i])
        
        plt.tight_layout()
        plt.show()

    def create_animation(self, interval=20):
        """创建波包演化动画"""
        Nx, Nt = self.B.shape
        
        fig = plt.figure(figsize=(10, 6))
        plt.axis([0, Nx, 0, np.max(self.V)*1.1])
        
        # 添加包含势垒参数的标题
        plt.title(f'量子隧穿动画 - 势垒宽度: {self.barrier_width}, 势垒高度: {self.barrier_height}', 
                 fontsize=12, fontweight='bold')
        plt.xlabel('位置')
        plt.ylabel('概率密度 / 势能')
        
        myline, = plt.plot([], [], 'r', lw=2, label=r'$|\psi|^2$')
        myline1, = plt.plot(self.x, self.V, 'k', lw=2, 
                           label=f'势垒 (宽度={self.barrier_width}, 高度={self.barrier_height})')
        
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3)
        
        def animate(i):
            myline.set_data(self.x, np.abs(self.B[:, i]))
            myline1.set_data(self.x, self.V)
            return myline, myline1
        
        anim = animation.FuncAnimation(fig, animate, frames=Nt, interval=interval)
        return anim

    def verify_probability_conservation(self):
        """验证演化过程中的概率守恒"""
        total_prob = np.zeros(self.Nt)
        for t in range(self.Nt):
            total_prob[t] = np.sum(np.abs(self.B[:, t])**2)
        
        return total_prob

    def demonstrate(self):
        """量子隧穿的演示函数"""
        print("量子隧穿模拟")
        print("=" * 40)
        
        # 求解方程
        print("解薛定谔方程中...")
        self.solve_schrodinger()
        T, R = self.calculate_coefficients()
        
        print(f"\n势垒宽度:{self.barrier_width}, 势垒高度:{self.barrier_height} 结果")
        print(f"透射系数: {T:.4f}")
        print(f"反射系数: {R:.4f}")
        print(f"总和 (T + R): {T + R:.4f}")
        
        # 绘制演化图
        print("绘制波函数演化图...")
        self.plot_evolution()
        
        # 检查概率守恒
        total_prob = self.verify_probability_conservation()
        print(f"概率守恒检查:")
        print(f"初始概率: {total_prob[0]:.6f}")
        print(f"最终概率: {total_prob[-1]:.6f}")
        print(f"相对变化: {abs(total_prob[-1] - total_prob[0])/total_prob[0]*100:.4f}%")
        
        # 创建动画
        print("创建动画...")
        anim = self.create_animation()
        plt.show()
        
        return anim


def demonstrate_quantum_tunneling():
    """便捷的演示函数"""
    solver = QuantumTunnelingSolver()
    return solver.demonstrate()


if __name__ == "__main__":
    # 运行演示
    barrier_width = 3
    barrier_height = 1.0
    solver = QuantumTunnelingSolver(barrier_width=barrier_width, barrier_height=barrier_height)
    animation = solver.demonstrate()

    barrier_width = 3
    barrier_height = 1.0
    solver = QuantumTunnelingSolver(barrier_width=barrier_width, barrier_height=barrier_height)
    animation = solver.demonstrate()
