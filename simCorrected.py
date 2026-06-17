import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import Normalize
from matplotlib.widgets import Slider

# --- Физические параметры ---
H = 10.0           # Высота стакана
T_cold = 20.0      # Температура на поверхности
T_neutral = 50.0   # Нейтральная плавучесть
dt = 0.05          # Шаг времени

# Изменяемые параметры (стартовые значения)
T_hot_val = 80.0
k_heat_val = 0.5
gamma_val = 1.0
c_buoy_val = 0.8
C_heat = 1.0       # Эффективная теплоемкость (для честного расчета теплоты в Джоулях)

# --- Начальные условия ---
y = 0.0
v = 0.0
T_drop = 20.0
time_val = 0.0

# Начальные термодинамические координаты
current_P = 100.0 + (H - y) * 0.5   # Оценочное давление в кПа (100 кПа на поверхности)
current_V = 1.0 + T_drop * 0.002    # Оценочный объем в Литрах (тепловое расширение)

# История (ограниченный буфер для графиков)
max_history = 400
t_hist, y_hist, T_hist = [], [], []
P_hist, V_hist = [], []

# Переменные для расчета цикла
W_accum = 0.0
Qin_accum = 0.0
last_W = 0.0
last_Qin = 0.0
last_eff = 0.0
state_in_air = False  # Флаг для отслеживания фазы полета

# --- Настройка окна Дашборда ---
fig = plt.figure(figsize=(14, 8))
fig.canvas.manager.set_window_title('Полная симуляция двигателя Дарлинга')

# 1. Левый график (Стакан)
ax_beaker = fig.add_axes([0.05, 0.35, 0.1, 0.55])
ax_beaker.set_title('Стакан')
ax_beaker.set_xlim(0, 2)
ax_beaker.set_ylim(0, H)
ax_beaker.set_xticks([])
ax_beaker.set_ylabel('Высота, см')

# ИСПРАВЛЕНИЕ: origin='lower' чтобы горячий (красный) был на дне (y=0)
gradient = np.linspace(T_hot_val, T_cold, 100).reshape(100, 1)
grad_img = ax_beaker.imshow(gradient, extent=[0, 2, 0, H], aspect='auto', cmap='coolwarm', alpha=0.4, origin='lower')
drop_plot = ax_beaker.scatter([1], [y], s=400, edgecolor='black', zorder=5)
norm = Normalize(vmin=T_cold, vmax=100.0)
cmap = plt.get_cmap('coolwarm')
temp_text = ax_beaker.text(0.1, H - 0.5, '', fontsize=9, bbox=dict(facecolor='white', alpha=0.8))

# 2. Центральный график (Высота от времени)
ax_yt = fig.add_axes([0.22, 0.35, 0.35, 0.55])
ax_yt.set_title('График y(t)')
ax_yt.set_ylim(0, H)
ax_yt.set_xlabel('Время, с')
ax_yt.grid(True)
line_yt, = ax_yt.plot([], [], lw=2, color='darkblue')

# 3. Правый график (Термодинамический цикл P-V)
ax_cycle = fig.add_axes([0.65, 0.35, 0.3, 0.55])
ax_cycle.set_title('Рабочий цикл: P-V диаграмма')
ax_cycle.set_xlabel('Объем капли (V), Листры')
ax_cycle.set_ylabel('Давление (P), кПа')
ax_cycle.grid(True)

# ИСПРАВЛЕНИЕ: Жестко фиксируем оси графика P-V, чтобы он не прыгал
ax_cycle.set_xlim(1.0, 1.3)
ax_cycle.set_ylim(98.0, 107.0)

line_cycle, = ax_cycle.plot([], [], lw=2, color='orange')
cycle_text = ax_cycle.text(0.05, 0.95, '', transform=ax_cycle.transAxes, fontsize=10,
                           va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

# --- Ползунки управления (внизу окна) ---
ax_thot = fig.add_axes([0.15, 0.20, 0.3, 0.03])
ax_kheat = fig.add_axes([0.15, 0.15, 0.3, 0.03])
ax_gamma = fig.add_axes([0.60, 0.20, 0.3, 0.03])
ax_cbuoy = fig.add_axes([0.60, 0.15, 0.3, 0.03])

slider_thot = Slider(ax_thot, 'T_дна (°C)', 50.0, 120.0, valinit=T_hot_val)
slider_kheat = Slider(ax_kheat, 'Теплопередача (k)', 0.1, 2.0, valinit=k_heat_val)
slider_gamma = Slider(ax_gamma, 'Вязкость (γ)', 0.1, 3.0, valinit=gamma_val)
slider_cbuoy = Slider(ax_cbuoy, 'Архимед (c)', 0.1, 2.0, valinit=c_buoy_val)

def update_params(val):
    global T_hot_val, k_heat_val, gamma_val, c_buoy_val
    T_hot_val = slider_thot.val
    k_heat_val = slider_kheat.val
    gamma_val = slider_gamma.val
    c_buoy_val = slider_cbuoy.val
    new_gradient = np.linspace(T_hot_val, T_cold, 100).reshape(100, 1)
    grad_img.set_data(new_gradient)

slider_thot.on_changed(update_params)
slider_kheat.on_changed(update_params)
slider_gamma.on_changed(update_params)
slider_cbuoy.on_changed(update_params)

# --- Физический движок ---
def update_physics():
    global y, v, T_drop, time_val
    global W_accum, Qin_accum, last_W, last_Qin, last_eff, state_in_air
    global current_P, current_V

    # 1. Температура среды
    T_water = T_hot_val - (T_hot_val - T_cold) * (y / H)
    
    # 2. Теплообмен (Нагрев/Охлаждение)
    dQ_dt = k_heat_val * (T_water - T_drop)
    T_drop += dQ_dt * dt
    
    # ИСПРАВЛЕНИЕ: Считаем теплоту с учетом теплоемкости (в Джоулях)
    if dQ_dt > 0:
        Qin_accum += C_heat * dQ_dt * dt
        
    # 3. Динамика
    a = c_buoy_val * (T_drop - T_neutral) - gamma_val * v
    v += a * dt  # Сначала обновляем скорость
    dy = v * dt
    y += dy      # Затем обновляем координату

    # 4. Граничные условия
    if y > 0.1:  # Слегка оторвались от дна
        state_in_air = True 
        
    if y <= 0:
        y = 0.0
        if v < 0:
            v = 0.0  # Гасим скорость только при ударе сверху вниз
            
        if state_in_air:
            last_W = W_accum
            last_Qin = Qin_accum
            # ИСПРАВЛЕНИЕ: Модуль на случай обратных флуктуаций
            last_eff = (abs(last_W) / abs(last_Qin)) * 100 if last_Qin > 0 else 0
            W_accum = 0.0
            Qin_accum = 0.0
            state_in_air = False
            
    elif y >= H:
        y = H
        if v > 0:
            v = 0.0
        
    time_val += dt
    
    # 5. Термодинамика P-V и расчет Работы
    prev_P = current_P
    prev_V = current_V
    
    current_P = 100.0 + (H - y) * 0.5  # Давление растет с глубиной (кПа)
    current_V = 1.0 + T_drop * 0.002   # Объем растет с температурой (Литры)
    
    # ИСПРАВЛЕНИЕ: Интегрирование честной работы dW = P * dV
    dV = current_V - prev_V
    dW = ((current_P + prev_P) / 2.0) * dV # Площадь трапеции для точности (кПа * Л = Дж)
    W_accum += dW
    
    # Запись в историю
    t_hist.append(time_val)
    y_hist.append(y)
    T_hist.append(T_drop)
    P_hist.append(current_P)
    V_hist.append(current_V)
    
    if len(t_hist) > max_history:
        t_hist.pop(0)
        y_hist.pop(0)
        T_hist.pop(0)
        P_hist.pop(0)
        V_hist.pop(0)

# --- Анимация ---
def animate(frame):
    for _ in range(4):
        update_physics()
        
    drop_plot.set_offsets(np.c_[1, y])
    drop_plot.set_facecolor(cmap(norm(T_drop)))
    temp_text.set_text(f'T среды: {T_hot_val - (T_hot_val - T_cold)*(y/H):.1f}°C\nT капли: {T_drop:.1f}°C')
    
    line_yt.set_data(t_hist, y_hist)
    if time_val > 20:
        ax_yt.set_xlim(time_val - 20, time_val)
    else:
        ax_yt.set_xlim(0, 20)
        
    line_cycle.set_data(V_hist, P_hist)
    
    cycle_text.set_text(
        f'Последний цикл:\n'
        f'Подведено тепла (Q): {last_Qin:.2f} Дж\n'
        f'Работа (A): {last_W:.2f} Дж\n'
        f'КПД: {last_eff:.2f}%'
    )
    
    return drop_plot, temp_text, line_yt, line_cycle, cycle_text

ani = animation.FuncAnimation(fig, animate, interval=20, blit=False, cache_frame_data=False)

plt.show()