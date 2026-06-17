import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import Normalize
from matplotlib.widgets import Slider

# --- Физические параметры ---
H = 0.2            # Высота стакана
T_cold = 20.0      # Температура на поверхности
T_neutral = 50.0   # Нейтральная плавучесть
dt = 0.05          # Шаг времени

# Изменяемые параметры (стартовые значения)
T_hot_val = 80.0
k_heat_val = 0.5
gamma_val = 1.0
c_buoy_val = 0.8

# --- Начальные условия ---
y = 0.0
v = 0.0
T_drop = 20.0
time_val = 0.0

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

# 3. Правый график (Термодинамический цикл T от y)
ax_cycle = fig.add_axes([0.65, 0.35, 0.3, 0.55])
ax_cycle.set_title('Рабочий цикл: P-V диаграмма')
ax_cycle.set_xlabel('Объем капли (V), отн. ед.')
ax_cycle.set_ylabel('Давление (P), кПа')
ax_cycle.grid(True)
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

    # 1. Температура среды
    T_water = T_hot_val - (T_hot_val - T_cold) * (y / H)

    # Эмпирические формулы плотности (кг/м^3)
    rho_water = 998.0 - 0.4 * (T_water - 20.0)
    rho_aniline = 1022.0 - 1.2 * (T_drop - 20.0)

    # Геометрия капли (пусть радиус будет 5 мм = 0.005 м)
    R = 0.005
    V_drop_m3 = (4.0 / 3.0) * 3.14159 * R**3
    S_cross = 3.14159 * R**2      # Площадь поперечного сечения (для трения)
    S_surf = 4.0 * 3.14159 * R**2 # Площадь поверхности (для теплообмена)
    mass = rho_aniline * V_drop_m3
    
# 2. Теплообмен (Закон Ньютона-Рихмана)
    C_p = 2000.0    # Удельная теплоемкость анилина, Дж/(кг*°C)
    h_conv = 1500.0 # Реальный коэффициент теплоотдачи в воде, Вт/(м2*°C)
    
    # Тепловой поток в Ваттах (Джоулях в секунду) через поверхность
    dQ_dt = h_conv * S_surf * (T_water - T_drop)
    
    # На сколько градусов нагреет этот поток нашу массу за время dt
    dT_drop = (dQ_dt / (mass * C_p)) * dt
    T_drop += dT_drop
    
    if dQ_dt > 0:
        Qin_accum += dQ_dt * dt  # Теперь это честные Джоули подведенного тепла!
        
    # 3. Динамика (через 2-й закон Ньютона)
    # Сила = Архимед (вверх) - Тяжесть (вниз)
    F_buoyancy = V_drop_m3 * 9.81 * rho_water
    F_gravity = mass * 9.81
    
    C_d = 0.47
    F_drag = -np.sign(v) * 0.5 * C_d * rho_water * S_cross * (v**2)
    
    F_net = F_buoyancy - F_gravity + F_drag
    a = F_net / mass
    
    v += a * dt  
    dy = v * dt
    y += dy      
    
    # Считаем работу как площадь P-V петли (A = P * dV)
    current_P = 100.0 + (H - y) * 0.5  # Давление в кПа
    dV = 0.002 * dT_drop               # Изменение объема (из формулы V = 1.0 + T_drop * 0.002)
    W_accum += current_P * dV          # Работа в Джоулях (кПа * Л)

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
            last_eff = (abs(last_W) / abs(last_Qin)) * 100 if last_Qin != 0 else 0
            W_accum = 0.0
            Qin_accum = 0.0
            state_in_air = False
            
    elif y >= H:
        y = H
        if v > 0:
            v = 0.0
        
    time_val += dt
    
# Давление растет с глубиной (H - y), Объем растет с температурой T_drop
    current_P = 100.0 + (H - y) * 0.5  # Оценочное давление в кПа
    current_V = 1.0 + T_drop * 0.002   # Оценочный объем (тепловое расширение)
    
    t_hist.append(time_val)
    y_hist.append(y)
    T_hist.append(T_drop)
    P_hist.append(current_P)  # Запись давления
    V_hist.append(current_V)  # Запись объема
    
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
        
    # Обновление P-V диаграммы
    line_cycle.set_data(V_hist, P_hist)
    
    # Динамическое масштабирование осей P-V (чтобы цикл всегда был в центре)
    if len(V_hist) > 0:
        ax_cycle.set_xlim(min(V_hist) - 0.01, max(V_hist) + 0.01)
        ax_cycle.set_ylim(min(P_hist) - 1, max(P_hist) + 1)
    
    cycle_text.set_text(
        f'Последний цикл:\n'
        f'Подведено тепла (Q): {last_Qin:.2f} у.е.\n'
        f'Работа (A): {last_W:.2f} у.е.\n'
        f'КПД: {last_eff:.1f}%'
    )
    
    return drop_plot, temp_text, line_yt, line_cycle, cycle_text

ani = animation.FuncAnimation(fig, animate, interval=20, blit=False, cache_frame_data=False)

plt.show()