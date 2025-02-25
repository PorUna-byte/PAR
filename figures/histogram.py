import matplotlib.pyplot as plt
import numpy as np

# gemma2-2b_llama3_winrates = [0.32, 0.64, 0.43, 0.44, 0.68, 0.62, 0.63, 0.68, 0.66]

import numpy as np
import matplotlib.pyplot as plt

def lab1():
    # 数据
    methods = ['Vanilla', 'WARM', 'ODIN', 'Reg', 'Meanstd', 'Clip', 'Minmax', 'LSC', 'PAR']
    gemma2_winrates = [0.29, 0.58, 0.35, 0.32, 0.33, 0.32, 0.61, 0.58, 0.63]

    # 设置条形图的位置和宽度
    x = np.arange(len(methods))  # 方法的标签位置
    width = 0.7  # 条形的宽度

    # 创建图形和轴
    fig, ax = plt.subplots(figsize=(12, 7))

    # 绘制条形图
    rects1 =  ax.bar(x, gemma2_winrates, width, label='Gemma2-2B', color='skyblue', alpha=0.8)

    # 添加标签、标题和图例
    ax.set_xlabel('Mitigaion Methods', fontsize=18)
    ax.set_ylabel('Average Winrate', fontsize=18)
    ax.set_title('Average Winrate', fontsize=18, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=0, ha='center', fontsize=18)  # 刻度标签水平显示
    ax.legend(fontsize=15)

    # 设置 y 轴范围
    ax.set_ylim(0.2, 0.7)

    # 在条形图上显示数值
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=18)

    autolabel(rects1)
    # autolabel(rects2)

    # 添加网格线
    ax.grid(True, linestyle='--', alpha=0.6)

    # 调整布局
    fig.tight_layout()

    # 保存为 PDF
    plt.savefig('gemma2-2b_ultrafb_bin_avgwin_lab1.pdf', format='pdf')

def lab5():
    # 数据
    methods = ['tanh', 'fittedpoly', 'sigmoid', 'sigmoidk2', 'sigmoidk3']
    centered_winrates = [0.62, 0.64, 0.63, 0.64, 0.65]
    uncentered_winrates = [0.59, 0.59, 0.59, 0.59, 0.57]

    # 设置条形图的位置和宽度
    x = np.arange(len(methods))  # 方法的标签位置
    width = 0.35  # 条形的宽度

    # 创建图形和轴
    fig, ax = plt.subplots(figsize=(12, 7))

    # 绘制条形图
    rects1 = ax.bar(x - width/2, centered_winrates, width, label='Centered', color='skyblue', alpha=0.8)
    rects2 = ax.bar(x + width/2, uncentered_winrates, width, label='Uncentered', color='lightcoral', alpha=0.8)

    # 添加标签、标题和图例
    ax.set_xlabel('Sigmoid-like function', fontsize=18)
    ax.set_ylabel('Average Winrate', fontsize=18)
    ax.set_title('Average Winrate', fontsize=18, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=0, ha='center', fontsize=18)  # 刻度标签水平显示
    ax.legend(fontsize=18)

    # 设置 y 轴范围
    ax.set_ylim(0.5, 0.7)

    # 在条形图上显示数值
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=18)

    autolabel(rects1)
    autolabel(rects2)

    # 添加网格线
    ax.grid(True, linestyle='--', alpha=0.6)

    # 调整布局
    fig.tight_layout()

    # 保存为 PDF
    plt.savefig('gemma2-2b_ultrafb_bin_avgwin_lab5.pdf', format='pdf')


if __name__=='__main__':
    lab1()
    lab5()