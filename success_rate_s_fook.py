import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import csv
import japanize_matplotlib

APPROACH_AREA = "right" # right or left

### 障害物距離読み込み ###
# csvファイル読み込み　(左右障害物と前後障害物はコメントアウトで切り替え)
df = pd.read_csv("/home/koki/ROBOMECH2026/data/obst_dist_left_and_right.csv", header=None)
# df = pd.read_csv("/home/koki/ROBOMECH2026/data/obst_dist_back_and_top.csv", header=None)

# 1,2列目のデータを取得して数値に変換
if APPROACH_AREA is "left":
  data = df[0].astype(float)
else:
  data = df[1].astype(float)

### 認識誤差 ###
mu=0
sigma=2 #誤差±2mm

plt.title(f"誤差±{sigma}[mm]のときの最適な側面のサイズ")
plt.xlabel("側面のサイズ[mm]") # x軸のラベル
plt.ylabel("アプローチ成功率") # y軸のラベル

graph_success_rate=[]
graph_s=[]

### 確率計算 ###
for s in range (5,30,1):
  # s = 13 # 側面のサイズ
  total = 0

  for i in range(len(data)):
    obst_dist = data[i]

    a = (obst_dist/2 - s/2) *-1
    b = (obst_dist/2 - s/2)

    # 積分
    area_cdf = norm.cdf(b, loc=mu, scale=sigma) - norm.cdf(a, loc=mu, scale=sigma)
    success_rate = max(area_cdf, 0)
    total += success_rate
    # print(f"No.{i} 区間 [{a}, {b}] の確率: {success_rate:.4f}, 左右障害物間距離{obst_dist_left + obst_dist_right}") # 各個体の成功率
  success_rate_average = total / len(data)
  graph_s.append(s)
  graph_success_rate.append(success_rate_average)
  print(f"誤差: {sigma}[mm], 側面サイズ: {s}[mm], そのときの成功率: {success_rate_average}")                   

plt.plot(graph_s, graph_success_rate)
plt.legend()
plt.show()

