import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import Normalize

# --- Физические параметры (подобраны для наглядности симуляции) ---
H = 10.0           # Высота стакана
T_hot = 80.0       # Температура на дне (песчаная баня)
T_cold = 20.0      # Температура на поверхности
T_neutral = 50.0   # Температура, при которой плотность анилина равна плотности воды
k_heat = 0.5       # Коэффициент теплопередачи
c_buoy = 0.8       # Коэффициент, связывающий разность температур с ускорением (Архимед)
gamma = 1.0        # Коэффициент вязкого трения
dt = 0.05          # Шаг интегрирования по времени

# --- Начальные условия ---
y = 0.0            # Капля начинает на дне
v = 0.0            # Начальная скорость
T_drop = 20.0      # Начальная температура капли (холодная)
time = 0.0

# Списки для хранения истории (для графика)
t_hist = []
y_hist = []
T_hist = []

# --- Настройка графики ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
fig.suptitle('Термодинамическая симуляция двигателя Дарлинга', fontsize=16)

# 1. Левый график (Стакан с водой)
ax1.set_title('Стакан (движение капли)')
ax1.set_xlim(0, 2)
ax1.set_ylim(0, H)
ax1.set_xticks([])
ax1.set_ylabel('Высота, см')

# Создаем фон градиента температуры воды
gradient = np.linspace(T_hot, T_cold, 100).reshape(100, 1)
ax1.imshow(gradient, extent=[0, 2, 0, H], aspect='auto', cmap='coolwarm', alpha=0.4)

# Отрисовка капли
drop_plot = ax1.scatter([1], [y], s=500, edgecolor='black', zorder=5)
norm = Normalize(vmin=T_cold, vmax=T_hot)
cmap = plt.get_cmap('coolwarm')

# Текстовые метки
temp_text = ax1.text(0.1, H - 0.5, '', fontsize=10, bbox=dict(facecolor='white', alpha=0.7))

# 2. Правый график (Фазовый портрет / Высота от времени)
ax2.set_title('График высоты от времени')
ax2.set_xlim(0, 20)
ax2.set_ylim(0, H)
ax2.set_xlabel('Время, с')
ax2.set_ylabel('Высота капли, см')
ax2.grid(True)
line, = ax2.plot([], [], lw=2, color='darkblue')

# --- Функция физического шага (Euler method) ---
def update_physics():
    global y, v, T_drop, time
    
    # Температура воды на текущей высоте
    T_water = T_hot - (T_hot - T_cold) * (y / H)
    
    # 1. Теплообмен
    dT = k_heat * (T_water - T_drop) * dt
    T_drop += dT
    
    # 2. Динамика (Ускорение)
    # Если T_drop > T_neutral, капля всплывает (a > 0)
    a = c_buoy * (T_drop - T_neutral) - gamma * v
    
    v += a * dt
    y += v * dt
    
    # 3. Граничные условия (дно и поверхность)
    if y <= 0:
        y = 0
        v = 0 # Удар о дно
    elif y >= H:
        y = H
        v = 0 # Удар о поверхность
        
    time += dt
    
    # Сохраняем историю
    t_hist.append(time)
    y_hist.append(y)
    T_hist.append(T_drop)
    
    # Ограничиваем длину истории для графика, чтобы он "полз"
    if time > 20:
        ax2.set_xlim(time - 20, time)

# --- Функция обновления анимации ---
def animate(frame):
    # Делаем несколько физических шагов на один кадр отрисовки для плавности
    for _ in range(3):
        update_physics()
    
    # Обновляем позицию капли
    drop_plot.set_offsets(np.c_[1, y])
    # Обновляем цвет капли в зависимости от её температуры
    drop_plot.set_facecolor(cmap(norm(T_drop)))
    
    # Обновляем текст
    temp_text.set_text(f'T воды рядом: {T_hot - (T_hot - T_cold)*(y/H):.1f}°C\nT капли: {T_drop:.1f}°C')
    
    # Обновляем график
    line.set_data(t_hist, y_hist)
    
    return drop_plot, temp_text, line

# Запуск анимации
ani = animation.FuncAnimation(fig, animate, frames=200, interval=30, blit=False)

plt.tight_layout()
plt.show()