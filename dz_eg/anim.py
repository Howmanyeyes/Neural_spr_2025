import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation

# Исходное изображение и ядро
image = np.array([
    [1, 2, 0, 3, 1],
    [0, 1, 2, 3, 1],
    [1, 2, 1, 0, 0],
    [2, 1, 0, 1, 1],
    [0, 1, 2, 3, 2]
])
kernel = np.array([
    [1, 0, -1],
    [1, 0, -1],
    [1, 0, -1]
])

# Параметры свёртки
stride = 1
padding = 0  # Padding пока не применяется (для простоты визуализации)

# Подготовка
kh, kw = kernel.shape
h, w = image.shape
out_h = (h - kh + 2 * padding) // stride + 1
out_w = (w - kw + 2 * padding) // stride + 1
output_map = np.zeros((out_h, out_w))

# Получение региона и результата
def get_region_and_result(x, y):
    region = image[y:y+kh, x:x+kw]
    multiplied = region * kernel
    result = np.sum(multiplied)
    return region, multiplied, result

# Визуализация
fig = plt.figure(figsize=(12, 6))
grid = fig.add_gridspec(2, 3, height_ratios=[2, 1])

ax1 = fig.add_subplot(grid[0, 0])
ax2 = fig.add_subplot(grid[0, 1])
ax3 = fig.add_subplot(grid[1, :2])
ax4 = fig.add_subplot(grid[0, 2])

ax1.set_title("Входное изображение")
ax1.imshow(image, cmap='gray')
rect = patches.Rectangle((0, 0), kw, kh, linewidth=2, edgecolor='r', facecolor='none')
ax1.add_patch(rect)
ax1.axis('off')

table = ax2.table(cellText=[[""]*kw]*kh, loc='center', cellLoc='center')
ax2.set_title("Регион × Ядро")
ax2.axis('off')

text_result = ax3.text(0.5, 0.5, "", fontsize=16, ha='center', va='center')
ax3.set_title("Результат свёртки")
ax3.axis('off')

ax4.set_title("Выходной тензор")
output_im = ax4.imshow(output_map, vmin=-10, vmax=10)
ax4.axis('off')

positions = [(x, y) for y in range(out_h) for x in range(out_w)]

def update(frame):
    x, y = positions[frame]
    region, multiplied, result = get_region_and_result(x, y)
    output_map[y, x] = result

    rect.set_xy((x - 0.5, y - 0.5))

    for i in range(kh):
        for j in range(kw):
            val = f"{region[i, j]}×{kernel[i, j]}={multiplied[i, j]}"
            table[i, j].get_text().set_text(val)

    text_result.set_text(f"Сумма: {result}")
    output_im.set_data(output_map)

ani = FuncAnimation(fig, update, frames=len(positions), interval=1200, repeat=False)
plt.show()

# in_channels  | Входной формат              | Количество входных "слоёв"
# out_channels | Сколько признаков извлекать | Кол-во каналов на выходе
# kernel_size  | Размер окна внимания        | Чем больше, тем грубее
# stride       | Шаг при сканировании        | Чем больше, тем меньше выход
# padding      | Добавляет границы           | Помогает сохранить размер
# dilation     | Расширяет ядро, делает дыры | Захватывает более далёкий контекст
