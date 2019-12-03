import numpy as np
import pandas as pd
from scipy.stats import chi2
import simplekml
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

########## USER INPUT ###########
log_file = "log_1200.csv"
#################################
log = np.loadtxt(log_file, delimiter=",", skiprows=1)
lat_array = log[:, 0]
lon_array = log[:, 1]

p = 0.997

mean_lat = np.mean(lat_array)
mean_lon = np.mean(lon_array)
print("mean", mean_lon, mean_lat)

cov = np.cov(lon_array, lat_array)  # 共分散行列
print("cov", cov)

vals, vecs = np.linalg.eigh(cov)  # 固有値・固有ベクトル
print("vals", vals)
print("vecs", vecs)

order = vals.argsort()[::-1]
print("order", order)
vals, vecs = vals[order], vecs[:, order]
print("vals", vals)
print("vecs", vecs)

theta = np.arctan2(*vecs[:,0][::-1])  # 座標変換の回転角
# theta = np.arctan2(vals[0] - vals[1], cov[0,1])  # 座標変換の回転角
print("theta", np.rad2deg(theta))

c = np.sqrt(chi2.ppf(p, 2))  # 自由度pのχ^2分布（マハラなんとかの距離）
a, b = c * np.sqrt(vals)  # 楕円の長軸短軸（地球半径の表示と同じa,b）
w = a * 2.0  # 楕円のwidth,height
h = b * 2.0
print("w", w)
print("h", h)

ell = Ellipse(xy=(mean_lon, mean_lat),
            width=w, height=h,
            angle=theta, color='black')
ell.set_facecolor('none')
ax = plt.subplot(111)
ax.add_artist(ell)
plt.scatter(lon_array, lat_array)
plt.show()

delta_angle = 1.0  # [deg]
angle_list = np.arange(0.0, 360.0, delta_angle)
ell_coords_lon = mean_lon + a * np.cos(np.deg2rad(angle_list))
ell_coords_lat = mean_lat + b * np.sin(np.deg2rad(angle_list))

kml = simplekml.Kml()
linestring = kml.newlinestring()
linestring.style.linestyle.color = simplekml.Color.orange
points = []
for i in range(len(ell_coords_lat)):
    points.append([ell_coords_lon[i], ell_coords_lat[i], 0.0])
linestring.coords = points
kml.save('./landing_ellipse.kml')


