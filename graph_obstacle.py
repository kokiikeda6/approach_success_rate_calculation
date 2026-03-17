import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import csv

plt.rcParams['font.family'] = 'Times New Roman' # Fonts


np.random.seed(1) #乱数を再現できるようにnp.random.seed(1)としておく
df_smpl=pd.DataFrame() #DataFrameに複数サンプルデータを格納する




### 障害物距離の分布 ###
# csvファイル読み込み 
label_obst="obst_dist"
df = pd.read_csv("/home/koki/ROBOMECH2026/data/obst_dist_left_and_right.csv", header=None)

# 1列目のデータを取得して数値に変換
data = df[0].astype(float)

data = data[data > 0] 
data_log = np.log(data)

# 平均と標準偏差を計算
data_mu = data_log.mean()
data_sigma = data_log.std()
print(f"Mean: {data_mu}")
print(f"Std Dev: {data_sigma}")
mu=data_mu
sigma=data_sigma
label_o="Obstacle distance"

count, bins, ignored = \
  plt.hist(df[0], bins=np.arange(0, 41, 1), density=True, align="mid", edgecolor='white', linewidth=1.0,
          # label=label_o,
          alpha =0.5)
x = np.linspace(1e-5, max(bins), 10000)
pdf_NL = (np.exp(-(np.log(x) - mu)**2 / (2 * sigma**2))
          / (np.sqrt(2 * np.pi) * sigma * x ))
# plt.plot(x, pdf_NL,color="blue",linestyle="dashed",)

# ymin, ymax = 0, 0.5
# plt.ylim(ymin,ymax)
plt.xlim([0,40])

# from matplotlib import font_manager
# fonts = font_manager.findSystemFonts(fontpaths=None, fontext='ttf')
# print(fonts)  # システム上の利用可能なTTFフォントのリストを表示


# plt.vlines(a, ymin, ymax, colors="red", linestyles="dashed", label="left: "+str(a)+"[mm]")
# plt.vlines(b, ymin, ymax, colors="blue", linestyles="dashed", label="right: "+str(b)+"[mm]")

# plt.title(f"success rate = {success_rate}")
plt.xlabel("Obstacle distance [mm]", fontsize=18) # x軸のラベル
plt.ylabel("Probability density", fontsize=18) # y軸のラベル

plt.xticks(fontsize=18)
plt.yticks(fontsize=18)

plt.legend()
plt.tight_layout()
plt.show()

