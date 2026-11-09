import pandas as pd
import matplotlib.pyplot as plt
from typing import List
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题

def BackTest_Display(df: pd.DataFrame, mark: str, other_marks: List[str]) -> None:
    """
    回测数据展示：
    - df: DataFrame['price', 'portfolio'] 原始价格 和 组合价格
    - mark: 策略名称
    """
    df = df / df.iloc[0]

    df['drawback'] = df['portfolio'] / df['portfolio'].cummax() - 1
    df['relative'] = df['portfolio'] / df['price']

    fig, ax1 = plt.subplots(figsize=(12, 6))

    ax1.plot(df.index, df['price'], label='Market', color='blue', linewidth=1)
    ax1.plot(df.index, df['portfolio'], label='Strategy', color='red', linewidth=1)
    ax1.plot(df.index, df['relative'], label='Relative', color='orange', linewidth=0.5)
    ax1.fill_between(df.index, df['relative'], 1, color='orange', alpha=0.3)

    for other_mark in other_marks:
        ax1.plot(df.index, df[other_mark], label=other_mark, linewidth=0.5)

    ax1.set_ylabel('value', fontsize=12)
    ax1.legend(loc='upper left')

    ax2 = ax1.twinx()

    ax2.plot(df.index, df['drawback'], label='Max DrawDown', color='red', linestyle='--', linewidth=0.1)
    ax2.fill_between(df.index, df['drawback'], df['drawback'].max(), color='red', alpha=0.3)

    ax2.set_ylim(-0.5, 0)

    ax2.set_ylabel('Max DrawDown (%)', fontsize=12)
    ax2.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(xmax=1))

    ax2.legend(loc='upper right')

    ax1.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))
    fig.autofmt_xdate()

    plt.title(f'Price VS {mark}', fontsize=14)
    plt.tight_layout()
    plt.show()

def ValuesComparation(df: pd.DataFrame, mark: str = None) -> None:
    """
    比较所有
    """
    plt.figure(figsize=(10, 6))

    for column in df.columns:
        if column == mark:
            plt.plot(df.index, df[column], label=column, color='red', linewidth=2, linestyle='--')  # mark
        else:
            plt.plot(df.index, df[column], label=column, alpha=0.7)

    # 设置边框隐藏
    ax = plt.gca()
    ax.spines['top'   ].set_visible(False)
    ax.spines['right' ].set_visible(False)
    ax.spines['left'  ].set_visible(False)
    ax.spines['bottom'].set_visible(True)

    plt.title('Comparation')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.show()

class SplitEffectViewer:
    def __init__(self, split_MatrixS: List[pd.DataFrame], factor_names: List[str] = None):
        self.current_plot_index = 0
        self.num_plots = len(split_MatrixS)
        self.split_MatrixS = split_MatrixS
        if factor_names != None:
            self.factor_names = factor_names
            self.with_names = True
        else: self.with_names = False
        self.fig, self.ax = plt.subplots()
        self.ax.spines['top'   ].set_visible(False)
        self.ax.spines['right' ].set_visible(False)
        self.ax.spines['left'  ].set_visible(False)
        self.ax.spines['bottom'].set_visible(True)
        self.fig.autofmt_xdate()
        self.update_plot()
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        plt.show()

    def update_plot(self):
        self.ax.clear()
        plot_Matrix = self.split_MatrixS[self.current_plot_index]
        for i, layer_mark in enumerate(plot_Matrix.columns):
            self.ax.plot(plot_Matrix.index, plot_Matrix[layer_mark], label=f'Layer {i+1}')
        
        if self.with_names: self.ax.set_title(f"Factor {self.factor_names[self.current_plot_index]} {self.current_plot_index + 1}/{self.num_plots}")
        else: self.ax.set_title(f"Factor {self.current_plot_index + 1}/{self.num_plots}")
        self.ax.set_xlabel("Time")
        self.ax.set_ylabel("Value")
        self.ax.legend()
        plt.xticks(rotation=45)
        plt.grid(True, axis='y')
        plt.tight_layout()
        plt.draw()

    def on_key(self, event):
        if event.key == 'right':
            self.current_plot_index = (self.current_plot_index + 1) % self.num_plots
            self.update_plot()
        elif event.key == 'left':
            self.current_plot_index = (self.current_plot_index - 1) % self.num_plots
            self.update_plot()
        elif event.key == 'escape':
            plt.close(self.fig)
        
def Correlation_Display(corr: pd.DataFrame, mark: str = None) -> None:
    """
    相关性矩阵
    """
    corr = corr.abs()
    plt.figure(figsize=(10, 6))
    if mark != None:
        plt.title(f'Correlation Matrix - {mark}')
    else:
        plt.title('Correlation Matrix')
    plt.imshow(corr, cmap='coolwarm', interpolation='nearest')
    plt.colorbar()
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90, fontsize=6)
    plt.yticks(range(len(corr.columns)), corr.columns, fontsize=6)
    plt.tight_layout()
    plt.show()

def plot_dual_y_axis(df, column1, column2):
    """
    绘制带有左右两个Y轴的图表。
    
    参数:
    df: 包含数据的DataFrame，其索引是时间。
    column1: 第一列数据的列名。
    column2: 第二列数据的列名。
    """
    # 创建一个图形和主轴
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # 绘制第一列数据
    color = 'tab:blue'
    ax1.set_xlabel('Time')
    ax1.set_ylabel(column1, color=color)
    ax1.plot(df.index, df[column1], color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    
    # 创建第二个Y轴
    ax2 = ax1.twinx()  # 实例化一个第二个Y轴
    color = 'tab:red'
    ax2.set_ylabel(column2, color=color)
    ax2.plot(df.index, df[column2], color=color)
    # ax2.plot(df.index, df['close'] / df['PE'], color='tab:green')

    ax2.tick_params(axis='y', labelcolor=color)
    
    # 设置标题
    plt.title(f'{column1} vs {column2}')
    
    # 显示图例
    fig.tight_layout()  # 调整布局
    fig.legend([column1, column2], loc='upper right', bbox_to_anchor=(1, 1))
    
    # 显示图形
    plt.show()

def plot_dataframe_columns(df:pd.DataFrame, figsize=(15, 10), title_prefix="Column", plot_type='line'):
    """
    为 Pandas DataFrame 的每一列绘制子图。

    参数:
        df (pd.DataFrame): 输入的 Pandas DataFrame。
        figsize (tuple): 图形的大小 (宽度, 高度)。
        title_prefix (str): 子图标题的前缀。
        plot_type (str): 绘图类型，支持 'line', 'bar', 'scatter' 等。
    """
    # 获取列的数量
    num_columns = len(df.columns)
    
    # 计算网格布局的行列数
    num_cols = 3  # 每行显示 3 个子图
    num_rows = num_columns // num_cols + (num_columns % num_cols > 0)
    
    # 创建网格布局
    fig, axs = plt.subplots(num_rows, num_cols, figsize=figsize)
    
    # 将 axs 转换为一维数组，方便遍历
    axs = axs.flatten()
    
    # 遍历每一列并绘制子图
    for i, col in enumerate(df.columns):
        ax = axs[i]
        
        if plot_type == 'line':
            ax.plot(df[col])
        elif plot_type == 'bar':
            ax.bar(df.index, df[col])
        elif plot_type == 'scatter':
            ax.scatter(df.index, df[col])
        else:
            raise ValueError("Unsupported plot type. Choose from 'line', 'bar', 'scatter'.")
        
        ax.set_title(f"{title_prefix} {col}")
        ax.set_xlabel("Index")
        ax.set_ylabel(col)
    
    # 隐藏多余的子图
    for i in range(num_columns, num_rows * num_cols):
        axs[i].axis('off')
    
    plt.tight_layout()  # 调整子图间距
    plt.show()

class DataFramePlotter:
    def __init__(self, df, columns_per_page=4, figsize=(15, 10), title_prefix="Column", plot_type='line'):
        """
        初始化 DataFramePlotter 类。

        参数:
            df (pd.DataFrame): 输入的 Pandas DataFrame。
            columns_per_page (int): 每页显示的列数，默认为 4。
            figsize (tuple): 图形的大小 (宽度, 高度)。
            title_prefix (str): 子图标题的前缀。
            plot_type (str): 绘图类型，支持 'line', 'bar', 'scatter' 等。
        """
        self.df = df
        self.columns_per_page = columns_per_page
        self.figsize = figsize
        self.title_prefix = title_prefix
        self.plot_type = plot_type
        self.num_columns = len(df.columns)
        self.num_pages = self.num_columns // columns_per_page + (self.num_columns % columns_per_page > 0)
        self.current_page = 0

        self.fig, self.axs = plt.subplots(2, 2, figsize=self.figsize)  # 每页 4 个子图
        self.axs = self.axs.flatten()

        # 初始绘制第一页
        self.redraw_page()

        self.fig.canvas.mpl_connect('key_press_event', self.on_key)

        # 显示图形
        plt.show()

    def on_key(self, event):
        if event.key == 'right':
            self.current_page = (self.current_page + 1) % self.num_pages
            self.redraw_page()
        elif event.key == 'left':
            self.current_page = (self.current_page - 1) % self.num_pages
            self.redraw_page()
        elif event.key == 'escape':
            plt.close(self.fig)

    def redraw_page(self):
        """重新绘制指定页的子图"""
        # 清除当前子图
        for ax in self.axs:
            ax.clear()

        # 计算当前页的列索引范围
        start_idx = self.current_page * self.columns_per_page
        end_idx = min(start_idx + self.columns_per_page, self.num_columns)

        # 绘制当前页的列
        for i, col in enumerate(self.df.columns[start_idx:end_idx]):
            ax = self.axs[i]
            if self.plot_type == 'line':
                ax.plot(self.df[col])
            elif self.plot_type == 'bar':
                ax.bar(self.df.index, self.df[col])
            elif self.plot_type == 'scatter':
                ax.scatter(self.df.index, self.df[col])
            else:
                raise ValueError("Unsupported plot type. Choose from 'line', 'bar', 'scatter'.")

            ax.set_title(f"{self.title_prefix} {col}")
            ax.set_xlabel("Index")
            ax.set_ylabel(col)

        # 隐藏多余的子图
        for i in range(end_idx - start_idx, self.columns_per_page):
            self.axs[i].axis('off')

        # 重新绘制图形
        self.fig.canvas.draw_idle()
