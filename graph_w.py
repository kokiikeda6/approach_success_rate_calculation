import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import csv


np.random.seed(1) #乱数を再現できるようにnp.random.seed(1)としておく
df_smpl=pd.DataFrame() #DataFrameに複数サンプルデータを格納する




### 障害物距離の分布 ###
# csvファイル読み込み 
label_obst="obst_dist"
df = pd.read_csv("/home/koki/ROBOMECH2026/data/obst_dist_ver2.csv", header=None)

# 1列目のデータを取得して数値に変換
data = df[0].astype(float)

data = data[data > 0] 
data_log = np.log(data)

# 平均と標準偏差を計算
# data_mu = data_log.mean()
# data_sigma = data_log.std()
# print(f"Mean: {data_mu}")
# print(f"Std Dev: {data_sigma}")
# mu=data_mu
# sigma=data_sigma
# label_o="Obstacle distance"

# count, bins, ignored = \
#   plt.hist(df, 50, density=True, align="mid",
#           label=label_o,
#           alpha =0.5)
# x = np.linspace(1e-5, max(bins), 10000)
# pdf_NL = (np.exp(-(np.log(x) - mu)**2 / (2 * sigma**2))
#           / (np.sqrt(2 * np.pi) * sigma * x ))
# plt.plot(x, pdf_NL,color="blue",linestyle="dashed",)


### 認識誤差 ###
mu=0
sigma=2
label_e="Errors arising from perception"
df_smpl[label_e] = np.random.normal(mu, sigma, 5000)
df_smpl
count, bins, ignored = \
  plt.hist(df_smpl[label_e], 50, density=True, align="mid",
          label=f"error distribution(μ={mu}, σ={sigma})", color="green",
          alpha =0.5)
x = np.linspace(min(bins), max(bins), 10000)
pdf_N = (np.exp(-(x - mu)**2 / (2 * sigma**2)) / (np.sqrt(2 * np.pi) * sigma ))
plt.plot(x, pdf_N,color="green",linestyle="dashed",)

ymin, ymax = 0, 0.3
plt.ylim(ymin,ymax)
plt.xlim([-10,10])

### エンドエフェクタサイズのlineを描画
# ee_size = 10
# plt.vlines(ee_size, ymin, ymax, colors="black", linestyles="dashed",
#            label="end-effector size: "+str(ee_size)+"[mm]")


### 確率計算 ###
w = 20 # 開口部のサイズ
p = 4 # 実験で得たパラメータ，開口部から-p[mm]
w_left = (w-4)/2
w_right = (w-4)/2

obst_dist_left = 30
obst_dist_right = 15

o_left = obst_dist_left - w/2
o_right = obst_dist_right - w/2
a = min(w_left, o_left) * -1
b = min(w_right, o_right)
# norm.cdf は -∞ から指定値までの面積（累積確率）を返します
# bまでの面積から、aまでの面積を引くことで区間の面積が出ます
area_cdf = norm.cdf(b, loc=mu, scale=sigma) - norm.cdf(a, loc=mu, scale=sigma)
success_rate = max(area_cdf, 0)
print(f"区間 [{a}, {b}] の確率: {success_rate:.4f}")

plt.vlines(a, ymin, ymax, colors="red", linestyles="dashed", label="left: "+str(a)+"[mm]")
plt.vlines(b, ymin, ymax, colors="blue", linestyles="dashed", label="right: "+str(b)+"[mm]")

plt.title(f"success rate = {success_rate}")
plt.xlabel("[mm]")                # x軸のラベル
plt.ylabel("Probability Density") # y軸のラベル

plt.legend()
plt.show()

