import os
import pandas as pd
import matplotlib.pyplot as plt
import plotly.io as pio
import logging
from tqdm import tqdm

def loss_plt(train_loss,plt_path):
    print('绘制并存储loss图像')
    # 创建一个新的图像
    plt.figure()
     # 绘制损失曲线
    plt.plot(train_loss)
    # 设置图像的标题、标签和图例
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    # 保存图像到指定路径
    plt.savefig(plt_path + 'loss.png')

def qlib_plt_loss(evals_result,model_name,save_path):
    print('绘制并存储loss图像')
    # Plot the loss curve
    plt.plot(evals_result)
    plt.xlabel("Iteration")
    plt.ylabel("Loss Value")
    plt.title("Model Training Loss Curve")
    plt.grid()
    plt.legend()
    
    # Save the loss curve to a specified folder
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, f"{model_name}.png"))
    # Display 
    #plt.show()
    pass

def report_plt(df,plt_path):# 将每一列绘制成一个单独的折线图
    # 为每一列绘制并保存图形
    for column in df.columns:
        plt.figure()  # 创建新图像
        plt.plot(df.index, df[column], marker='o', label=column)
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.title(f'Line Plot for {column}')
        plt.legend()
    
        # 保存图像到文件夹
        plt.savefig(plt_path + f"{column}_plot.png")
    #   关闭所有图像
    #plt.close('all')

def column_plt(df,plt_path):
    # 绘制柱状图
    ax = df.plot.bar(rot=0, figsize=(12, 6))

# 添加图表标题和轴标签
    ax.set_title('Comparison of Excess Returns with and without Cost')
    ax.set_xlabel('Strategies')
    ax.set_ylabel('Values')
    plt.legend()

    # 显示图表
    #plt.show()
    plt.savefig(plt_path + "column_plt.png")

logging.basicConfig(level=logging.INFO)

def plot_and_save_figures(fig_list, plt_path):
    """
    Plot and save a list of Plotly figures to a specified directory.
    
    Args:
        fig_list (list): A list of Plotly figure objects to be plotted and saved.
        plt_path (str): The directory path where the images will be saved.
    """
    renderer = None
    for idx, fig in enumerate(tqdm(fig_list)):

        logging.info(f"Processing figure {idx + 1} of {len(fig_list)}...")

        logging.info("Drawing the figure...")
        fig.show(renderer=renderer)

        # Create a unique file name for each figure
        output_filename = f"output_{idx}.png"
        
        # Use os.path.join to construct the file path
        output_filepath = os.path.join(plt_path, output_filename)
        logging.info("Saving the figure...")
        # io.write_image(fig, output_filepath, width=80, height=60, scale=0.2) #保存图像
        logging.info(f"Figure {idx + 1} processed.")

def compare_report_normal_df(folder_path):
    print('绘制并存储report_normal_df')
    # 读取文件夹内所有csv数据，存储为以文件名命名的dataframe格式数据
    data_dict = {}
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.csv'):
            file_path = os.path.join(folder_path, file_name)
            df = pd.read_csv(file_path,index_col=0,header=0)
            df = df.iloc[1:]
            data_dict[file_name] = df
    # 绘制所有dataframe数据里的相同列名的数据在一张折线图上
    cols = set.intersection(*[set(df.columns) for df in data_dict.values()])
    for col in cols:
        plt.figure()
        plt.title(col)
        for file_name, df in data_dict.items():
            if col in df.columns:
                plt.plot(df[col], label=file_name)
        plt.legend()
        plt.savefig(os.path.join(folder_path, f"{col}.png"))
        plt.show()
    pass

def bar_plot(with_cost_df,folder_path,df_name):
    fig,axes = plt.subplots(nrows=1, ncols=5, figsize=(20, 3))
    metrics = with_cost_df.index
    colormap = 'viridis'

    for i, metric in enumerate(metrics):
        ax = axes[i]
        # 绘制单个柱状图
        with_cost_df.loc[metric].plot.bar(ax=ax, colormap=colormap, alpha=0.8)
        # 添加标题和轴标签
        ax.set_title(f"{metric}", fontsize=14, fontweight='bold')
        ax.set_ylabel("Risk", fontsize=12)
        # 添加网格线
        ax.yaxis.grid(True, linestyle='--', linewidth=0.5)

        # 设置刻度字体大小
        ax.tick_params(axis='both', labelsize=10)

        # 调整子图间距
        plt.subplots_adjust(wspace=0.8)

    # 显示图像
    plt.legend()
    #plt.show()
    plt.savefig(os.path.join(folder_path+f"{df_name}.png"))
    pass

def compare_analysis_df(folder_path):
    print('绘制并存储analysis_df')
    # 获取文件夹内所有 "model_" 开头的 CSV 文件
    csv_files = [f for f in os.listdir(folder_path) if f.startswith("model_") and f.endswith(".csv")]

    model_name = []
    # 初始化一个空的 DataFrame
    df = pd.DataFrame()

    for file in csv_files:
        # 读取 CSV 文件为 DataFrame
        temp_df = pd.read_csv(os.path.join(folder_path, file), index_col=(0,1))
    
        # 获取文件名
        model_name = os.path.splitext(file)[0]

        # 将列名更改为文件名
        temp_df.columns = [model_name]

        # 将数据添加到空的 DataFrame
        df = pd.concat([df, temp_df], axis=1)

    # 根据 'excess_return_with_cost' 和 'excess_return_without_cost' 拆分数据
    with_cost_df = df.loc['excess_return_with_cost']
    without_cost_df = df.loc['excess_return_without_cost']
    bar_plot(with_cost_df,folder_path,'excess_return_with_cost')
    bar_plot(without_cost_df,folder_path,'excess_return_without_cost')

    pass

def run_plt():

    pass


