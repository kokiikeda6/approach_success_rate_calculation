import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import csv
import japanize_matplotlib

plt.rcParams['font.family'] = 'Times New Roman' # Fonts

### 障害物距離読み込み ###
# csvファイル読み込み　(左右障害物と前後障害物はコメントアウトで切り替え)
# df = pd.read_csv("/home/koki/ROBOMECH2026/data/obst_dist_left_and_right.csv", header=None) #左右方向の障害物
df = pd.read_csv("/home/koki/ROBOMECH2026/data/obst_dist_back_and_top.csv", header=None) # 前後方向の障害物

# 1,2列目のデータを取得して数値に変換
data_left = df[0].astype(float)
data_right = df[1].astype(float)

### 認識誤差 ###
mu=0
sigma=5 #誤差 ±xmm

# plt.title(f"誤差±{sigma}[mm]のときの最適な開口部のサイズ")

plt.title(f"Optimal opening size (σ = {sigma})", fontsize=18)
plt.xlabel("Opening size", fontsize=14) # x軸のラベル
plt.ylabel("Success rate", fontsize=14) # y軸のラベル

graph_success_rate=[]
graph_w=[]

### 確率計算 ###
for _ in range (5,6,1):
  w = 7 + 4 # 開口部のサイズ 果柄径の最大値+4mm
  ee_out_side = 0 # エンドエフェクタの厚み
  delta_z = +50 # 開口部25mmに対し，実効許容誤差 約76mm
  # w_left = w/2 - 2
  w_left = (w + delta_z) / 2
  # w_right = w/2 - 2
  w_right = (w + delta_z) / 2
  total = 0

  for i in range(len(data_left)):
    obst_dist_left = data_left[i]
    obst_dist_right = data_right[i]
    o_left = obst_dist_left - w/2 - ee_out_side
    o_right = obst_dist_right - w/2 - ee_out_side

    a = min(w_left, o_left) *-1
    b = min(w_right, o_right)

    # 積分
    area_cdf = norm.cdf(b, loc=mu, scale=sigma) - norm.cdf(a, loc=mu, scale=sigma)
    success_rate = max(area_cdf, 0)
    total += success_rate
    print(f"No.{i} 区間 [{a}, {b}] の確率: {success_rate:.4f}, 左右障害物間距離{obst_dist_left + obst_dist_right}") # 各個体の成功率
  success_rate_average = total / len(data_left)
  graph_w.append(w)
  graph_success_rate.append(success_rate_average)
  print(f"誤差: {sigma}[mm], 開口部サイズ: {w}[mm], そのときの成功率: {success_rate_average}")                   

plt.plot(graph_w, graph_success_rate)
plt.ylim([0.0,0.45])


plt.legend()
plt.show()

