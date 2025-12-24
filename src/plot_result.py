import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker

def plot_benchmark(csv_path='benchmark_results.csv'):
    # 1. データの読み込み
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: {csv_path} が見つかりません。")
        return

    # 失敗したタスクを除外
    df_success = df[df['status'] == 'Success'].copy()

    # 2. 指定カラーパレットの定義
    # データ内の library 名と完全に一致させる必要があります
    custom_palette = {
        'Pandas': 'red',           # 赤
        'Polars(cpu)': 'navy',     # 濃い青
        'Polars(gpu)': 'deepskyblue',     # 濃い水色
        'cuDF': 'green',           # 緑
        'DuckDB': 'gold',          # 黄 (純粋なyellowは見にくいのでgoldを使用)
        'Dask(CPU)': 'hotpink',    # ピンク
        'Spark': 'orange'          # オレンジ
    }

    # データに含まれていないライブラリがある場合のエラー回避（デフォルト色を割り当て）
    libraries = df_success['library'].unique()
    for lib in libraries:
        if lib not in custom_palette:
            custom_palette[lib] = 'gray'

    # スタイル設定
    sns.set_theme(style="whitegrid")
    
    tasks = df_success['task'].unique()
    tasks.sort()

    # サブプロット作成
    n_cols = 3
    n_rows = (len(tasks) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 12), sharex=True)
    axes = axes.flatten()

    # 3. 描画ループ
    for i, task in enumerate(tasks):
        ax = axes[i]
        task_data = df_success[df_success['task'] == task]

        sns.lineplot(
            data=task_data,
            x='rows',
            y='duration_sec',
            hue='library',
            style='library',
            palette=custom_palette,  # ここで色を指定
            markers=True,
            dashes=False,
            ax=ax,
            linewidth=2.5,  # 線を少し太く
            markersize=9
        )

        ax.set_title(task, fontsize=14, fontweight='bold')
        ax.set_ylabel('Duration (sec) [Log Scale]')
        ax.set_xlabel('Rows [Log Scale]')

        # 両対数グラフ設定
        ax.set_xscale('log')
        ax.set_yscale('log')

        # X軸フォーマット (1M, 10M, 1B)
        def format_xaxis(x, pos):
            if x >= 1e9: return f'{int(x/1e9)}B'
            if x >= 1e6: return f'{int(x/1e6)}M'
            if x >= 1e3: return f'{int(x/1e3)}K'
            return str(int(x))
            
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_xaxis))
        ax.grid(True, which="both", ls="--", alpha=0.5)

        # 個別の凡例は削除
        if ax.get_legend():
            ax.get_legend().remove()

    # 余白の削除
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # 4. 共通の凡例を作成
    # 凡例の順序を固定したい場合は hue_order を使う手もありますが、ここでは既存のハンドルから生成
    handles, labels = axes[0].get_legend_handles_labels()
    
    # 凡例も指定した色で見やすく配置
    fig.legend(
        handles, 
        labels, 
        loc='upper center', 
        bbox_to_anchor=(0.5, 1.05), 
        ncol=len(libraries), 
        fontsize=12,
        title="Library"
    )

    plt.tight_layout()
    
    output_file = 'benchmark_results.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"グラフを保存しました: {output_file}")

if __name__ == "__main__":
    plot_benchmark()