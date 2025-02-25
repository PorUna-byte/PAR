import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# 定义 Sigmoid 函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoidk2(x):
    return 1 / (1 + np.exp(-2*x))

def sigmoidk3(x):
    return 1 / (1 + np.exp(-3*x))

def identity(x):
    return x

def sgfc(x):
    """
    自定义函数：
    - x <= 0: 值为 0.5
    - 0 < x <= 2: 缓慢增长
    - 2 < x <= 3: 快速增长至 0.9
    - x > 3: 逐渐收敛到 1
    """
    return sigmoidk3(x-3)
    
# 定义 tanh 函数
def tanh(x):
    return np.tanh(x)

# 定义高阶多项式函数
def high_order_poly(x, a, b, c, d, e, f):
    return a * x**5 + b * x**4 + c * x**3 + d * x**2 + e * x + f

# 生成 Sigmoid 数据用于拟合
x_fit = np.linspace(-5, 5, 100)
y_fit = sigmoid(x_fit)

# 使用 curve_fit 进行多项式拟合
params, _ = curve_fit(high_order_poly, x_fit, y_fit)

# 定义新分段多项式函数
def poly_fitted(x):
    return np.piecewise(
        x,
        [x <= -5, (x > -5) & (x < 5), x >= 5],
        [0, lambda x: high_order_poly(x, *params), 1]
    )

# 生成数据
x = np.linspace(-5, 5, 500)
y_sigmoid = sigmoid(x)
y_sigmoidk2 = sigmoidk2(x)
y_sigmoidk3 = sigmoidk3(x)
y_tanh = tanh(x)
y_polyfitted = poly_fitted(x)
y_identity = identity(x)
y_sgfc = sgfc(x)

# 绘制图像
plt.figure(figsize=(12, 7))

# 绘制不同函数
plt.plot(x, y_sigmoid, label="sigmoid(x)", color="blue", linewidth=2)
plt.plot(x, y_sigmoidk2, label="sigmoidk2(x)", color="green", linewidth=2)
plt.plot(x, y_sigmoidk3, label="sigmoidk3(x)", color="orange", linewidth=2)
plt.plot(x, y_tanh, label="tanh(x)", color="red", linewidth=2)
plt.plot(x, y_polyfitted, label="poly_fitted(x)", color="purple", linestyle="--", linewidth=2)
plt.plot(x, y_sgfc, label='sgfc(x)', color='yellow', linewidth=2)
plt.plot(x, y_identity, label="identity(x)", color="black", linewidth=2)

# 添加标题和标签
plt.title("Sigmoid-like Functions", fontsize=15, pad=20)
plt.xlabel("Centered Reward", fontsize=15, labelpad=10)
plt.ylabel("RL Reward", fontsize=15, labelpad=10)

# 设置网格
plt.grid(True, linestyle="--", alpha=0.6)

# 设置图例
plt.legend(fontsize=18, loc='lower right')

# 设置坐标轴范围
plt.xlim(-5, 5)
plt.ylim(-1.2, 1.2)

# 调整布局
plt.tight_layout()

# 保存图像
plt.savefig('sigmoid_like_function.pdf', bbox_inches='tight')

# 显示图像
plt.show()