import matplotlib.pyplot as plt
from matplotlib import image

squares = [1, 4, 9, 16]
plt.plot(squares, linewidth=5)
# 设置图标标题、并给坐标轴加上标签
plt.title("Square Numbers:", fontsize=23)
plt.xlabel("Value", fontsize=14)
plt.ylabel("Square Of Value", fontsize=14)
# 设置刻度标记的大小
plt.tick_params(axis='both', labelsize=10)

plt.close()

# scatter plot
x_values = list(range(1, 10))
y_values = [x ** 2 for x in x_values]

# 调整图像尺寸大小
plt.figure(figsize=(10, 5))
plt.scatter(x_values, y_values, s=40, edgecolors='red', c=y_values, cmap=plt.cm.Blues)

# 设置每个坐标轴的取值范围(函数axis()要求提供四个值：x和y坐标轴的最小值和最大值)
plt.axis([0, 20, 0, 500])

# 隐藏坐标轴
plt.axes().get_xaxis().set_visible(True)
plt.axes().get_yaxis().set_visible(True)

plt.show()

# 保存图像
# plt.savefig('t.png', bbox_inches='tight')

if __name__ == '__main__':
    pass
