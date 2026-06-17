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
R_mm_val = 5.0        # Радиус капли в миллиметрах
h_conv_val = 1500.0   # Коэффициент теплоотдачи Вт/(м2*°C)

# --- Начальные условия ---
y = 0.0
v = 0.0
T_drop = 20.0
time_val = 0.0

# История (ограниченный буфер для графиков)
max_history = 400
t_hist, y_hist, T_hist = [], [], []
P_hist, V_hist = [], []
current_P = 101.325
current_V = 0.0

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

fig.text(0.02, 0.90, 
         "СИСТЕМА: Термодинамический двигатель Дарлинга\n"
         "Среда: Вода (P_атм = 101.3 кПа, Ускорение g = 9.81 м/с²)\n"
         "Рабочее тело: Жидкий анилин (Теплоемкость C_p = 2000 Дж/кг·К)\n"
         "Механизм: Периодическое изменение плавучести из-за теплового расширения", 
         fontsize=10, bbox=dict(facecolor='#f8f9fa', alpha=0.9, edgecolor='gray'))

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
temp_text = ax_beaker.text(0.1, H * 0.7, '', fontsize=9, bbox=dict(facecolor='white', alpha=0.8))

# 2. Центральный график (Высота от времени)
ax_yt = fig.add_axes([0.22, 0.35, 0.35, 0.55])
ax_yt.set_title('График y(t)')
ax_yt.set_ylim(0, H * 100)
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
ax_rad = fig.add_axes([0.15, 0.15, 0.3, 0.03])
ax_hconv = fig.add_axes([0.60, 0.20, 0.3, 0.03])

slider_thot = Slider(ax_thot, 'T_дна (°C)', 50.0, 100.0, valinit=T_hot_val)
slider_rad = Slider(ax_rad, 'Радиус R (мм)', 2.0, 10.0, valinit=R_mm_val)
slider_hconv = Slider(ax_hconv, 'Теплоотдача h (Вт/м²·К)', 500.0, 4000.0, valinit=h_conv_val)

def update_params(val):
    global T_hot_val, R_mm_val, h_conv_val
    T_hot_val = slider_thot.val
    R_mm_val = slider_rad.val
    h_conv_val = slider_hconv.val
    # Обновляем градиент фона
    new_gradient = np.linspace(T_hot_val, T_cold, 100).reshape(100, 1)
    grad_img.set_data(new_gradient)

slider_thot.on_changed(update_params)
slider_rad.on_changed(update_params)
slider_hconv.on_changed(update_params)

# --- Физический движок ---
def update_physics():
    global y, v, T_drop, time_val
    global W_accum, Qin_accum, last_W, last_Qin, last_eff, state_in_air
    global current_P, current_V, mass

    # 1. Температура среды
    T_water = T_hot_val - (T_hot_val - T_cold) * (y / H)

    # Эмпирические формулы плотности (кг/м^3)
    rho_water = 998.0 - 0.4 * (T_water - 20.0)
    rho_aniline = 1022.0 - 1.2 * (T_drop - 20.0)

    # Геометрия капли (Масса постоянна, объем дышит от температуры)
    R_base = R_mm_val / 1000.0  
    V_base = (4.0 / 3.0) * 3.14159 * R_base**3
    mass = 1022.0 * V_base  # Базовая масса при 20°C фиксируется
    
    # ТЕПЕРЬ ОБЪЕМ МЕНЯЕТСЯ ОТ НАГРЕВА!
    V_drop_m3 = mass / rho_aniline
    
    # Актуальный радиус для расчета трения и площади теплообмена
    R_current = (0.75 * V_drop_m3 / 3.14159)**(1/3) 
    S_cross = 3.14159 * R_current**2      
    S_surf = 4.0 * 3.14159 * R_current**2
    
    # 2. Теплообмен (Закон Ньютона-Рихмана)
    C_p = 2000.0    
    h_conv = h_conv_val    
    
    dQ_dt = h_conv * S_surf * (T_water - T_drop)
    dT_drop = (dQ_dt / (mass * C_p)) * dt
    T_drop += dT_drop
    
    if dQ_dt > 0:
        Qin_accum += dQ_dt * dt  # Честные Джоули
        
    # 3. Динамика (через 2-й закон Ньютона)
    F_buoyancy = V_drop_m3 * 9.81 * rho_water
    F_gravity = mass * 9.81
    C_d = 0.47
    F_drag = -np.sign(v) * 0.5 * C_d * rho_water * S_cross * (v**2)
    
    F_net = F_buoyancy - F_gravity + F_drag
    a = F_net / mass
    
    v += a * dt  
    y += v * dt      
    
    # 4. Граничные условия
    if y > 0.005:  # <-- ИСПРАВЛЕНО НА 5 мм
        state_in_air = True 
        
    if y <= 0:
        y = 0.0
        if v < 0: v = 0.0 
            
        if state_in_air:
            last_W = W_accum
            last_Qin = Qin_accum
            last_eff = (abs(last_W) / abs(last_Qin)) * 100 if last_Qin != 0 else 0
            W_accum = 0.0
            Qin_accum = 0.0
            state_in_air = False
            
    elif y >= H:
        y = H
        if v > 0: v = 0.0
        
    time_val += dt
    
    # 5. Термодинамика P-V и расчет Работы
    prev_P = current_P
    prev_V = current_V
    
    # Честное давление: Атмосферное (101.3 кПа) + Гидростатика (кПа)
    current_P = 101.325 + (rho_water * 9.81 * (H - y)) / 1000.0 
    
    # Объем капли в миллилитрах (мл = см^3)
    current_V = V_drop_m3 * 1e6   
    
    # Работа: dW = P * dV. кПа * мл = 10^-3 Дж (Миллиджоули)
    dV_ml = current_V - prev_V
    dW_Joules = ((current_P + prev_P) / 2.0) * dV_ml * 1e-3
    W_accum += dW_Joules
    
    # Запись в историю
    t_hist.append(time_val)
    y_hist.append(y * 100) # для графика высоты переводим в см
    T_hist.append(T_drop)
    P_hist.append(current_P)
    V_hist.append(current_V)

    # Ограничение длины истории (стираем старые хвосты)
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

    drop_plot.set_sizes([(R_mm_val / 5.0)**2 * 400])

    # Выводим исчерпывающее состояние системы
    temp_text.set_text(
        f'--- Текущее состояние ---\n'
        f'Масса: {mass*1000:.2f} г\n'
        f'Объем: {current_V:.2f} мл\n'
        f'T капли: {T_drop:.1f} °C\n'
        f'T воды (вокруг): {T_hot_val - (T_hot_val - T_cold)*(y/H):.1f} °C'
    )
    
    line_yt.set_data(t_hist, y_hist)
    if time_val > 20:
        ax_yt.set_xlim(time_val - 20, time_val)
    else:
        ax_yt.set_xlim(0, 20)
        
    # Обновление P-V диаграммы
    line_cycle.set_data(V_hist, P_hist)
    
    # Динамическое масштабирование осей P-V (чтобы цикл всегда был в центре)
    if len(V_hist) > 0:
        v_min, v_max = min(V_hist), max(V_hist)
        dv = (v_max - v_min) * 0.1 if v_max != v_min else 0.1
        ax_cycle.set_xlim(v_min - dv, v_max + dv)
        ax_cycle.set_ylim(101.0, 103.5) # Давление всегда в диапазоне 101.3 - 103.3 кПа
    
    cycle_text.set_text(
        f'Последний цикл:\n'
        f'Подведено тепла (Q): {last_Qin:.2f} Дж\n'
        f'Работа (A): {last_W * 1e6:.2f} мкДж\n'
        f'КПД: {last_eff:.5f}%'
    )
    
    return drop_plot, temp_text, line_yt, line_cycle, cycle_text

ani = animation.FuncAnimation(fig, animate, interval=20, blit=False, cache_frame_data=False)

plt.show()