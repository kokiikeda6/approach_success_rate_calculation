import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import csv
import japanize_matplotlib

### 障害物距離読み込み ###
# csvファイル読み込み
# df = pd.read_csv("/home/koki/ROBOMECH2026/data/obst_dist_left_and_side.csv", header=None)
df = pd.read_csv("/home/koki/ROBOMECH2026/data/obst_dist_back_and_top.csv", header=None)

# 1,2列目のデータを取得して数値に変換
data_left = df[0].astype(float)
data_right = df[1].astype(float)

### 認識誤差 ###
mu=0
sigma=2 #誤差±2mm

plt.title(f"誤差±{sigma}[mm]のときの最適な側面のサイズ")
plt.xlabel("側面のサイズ[mm]")                # x軸のラベル
plt.ylabel("アプローチ成功率") # y軸のラベル

graph_success_rate=[]
graph_s=[]

### 確率計算 ###
for s in range (5,30,1):
  # s = 13 # 側面のサイズ
  total = 0
  peduncle_size = 5
  s_left = (s - peduncle_size)/2
  s_right = (s - peduncle_size)/2

  for i in range(len(data_left)):
    obst_dist_left = data_left[i]
    obst_dist_right = data_right[i]
    o_left = obst_dist_left - s/2
    o_right = obst_dist_right - s/2

    a = min(s_left, o_left) *-1
    b = min(s_right, o_right)

    # 積分
    area_cdf = norm.cdf(b, loc=mu, scale=sigma) - norm.cdf(a, loc=mu, scale=sigma)
    success_rate = max(area_cdf, 0)
    total += success_rate
    # print(f"No.{i} 区間 [{a}, {b}] の確率: {success_rate:.4f}, 左右障害物間距離{obst_dist_left + obst_dist_right}") # 各個体の成功率
  success_rate_average = total / len(data_left)
  graph_s.append(s)
  graph_success_rate.append(success_rate_average)
  print(f"誤差: {sigma}[mm], 側面サイズ: {s}[mm], そのときの成功率: {success_rate_average}")                   

plt.plot(graph_s, graph_success_rate)
plt.legend()
plt.show()

