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

max_history = 400
t_hist, y_hist, T_hist = [], [], []
P_hist, V_hist = [], []
current_P = 101.325
current_V = 0.0

W_accum, Qin_accum = 0.0, 0.0
last_W, last_Qin, last_eff = 0.0, 0.0, 0.0
state_in_air = False

# =============================================================================
# --- НАСТРОЙКА ИНТЕРФЕЙСА (ДАШБОРД) ---
# =============================================================================
fig = plt.figure(figsize=(15, 9))
fig.canvas.manager.set_window_title('Двигатель Дарлинга: Физическая симуляция')
fig.patch.set_facecolor('#e9ecef') # Светло-серый фон всего окна

# --- ВЕРХНЯЯ ЗОНА: ГРАФИКИ (y = 0.45 ... 0.95) ---

# 1. Стакан
ax_beaker = fig.add_axes([0.05, 0.45, 0.10, 0.45])
ax_beaker.set_title('Стакан (20 см)', fontweight='bold')
ax_beaker.set_xlim(0, 2)
ax_beaker.set_ylim(0, H)
ax_beaker.set_xticks([])
ax_beaker.set_ylabel('Высота, см')
gradient = np.linspace(T_hot_val, T_cold, 100).reshape(100, 1)
grad_img = ax_beaker.imshow(gradient, extent=[0, 2, 0, H], aspect='auto', cmap='coolwarm', alpha=0.4, origin='lower')
drop_plot = ax_beaker.scatter([1], [y], s=400, edgecolor='black', zorder=5)
norm = Normalize(vmin=T_cold, vmax=100.0)
cmap = plt.get_cmap('coolwarm')

# 2. График Высоты
ax_yt = fig.add_axes([0.22, 0.45, 0.35, 0.45])
ax_yt.set_title('График высоты y(t)', fontweight='bold')
ax_yt.set_ylim(0, H * 100)
ax_yt.set_xlabel('Время, с')
ax_yt.grid(True, linestyle='--', alpha=0.7)
line_yt, = ax_yt.plot([], [], lw=2.5, color='#2b8a3e')

# 3. График P-V цикла
ax_cycle = fig.add_axes([0.65, 0.45, 0.30, 0.45])
ax_cycle.set_title('Термодинамический цикл P-V', fontweight='bold')
ax_cycle.set_xlabel('Объем капли V, мл')
ax_cycle.set_ylabel('Давление P, кПа')
ax_cycle.grid(True, linestyle='--', alpha=0.7)
line_cycle, = ax_cycle.plot([], [], lw=2.5, color='#e67700')

# --- НИЖНЯЯ ЗОНА: ПАНЕЛЬ УПРАВЛЕНИЯ (y = 0.05 ... 0.35) ---

# Общий стиль для плашек
bbox_style = dict(boxstyle='round,pad=1', facecolor='white', edgecolor='#ced4da', alpha=0.95)

# Левая колонка: Текст состояния системы
ax_text_state = fig.add_axes([0.05, 0.05, 0.25, 0.30])
ax_text_state.axis('off') # Прячем оси, оставляем только текст
temp_text = ax_text_state.text(0.0, 0.5, '', fontsize=11, va='center', linespacing=1.6, bbox=bbox_style)

# Центральная колонка: Ползунки
ax_thot  = fig.add_axes([0.40, 0.25, 0.20, 0.03])
ax_rad   = fig.add_axes([0.40, 0.15, 0.20, 0.03])
ax_hconv = fig.add_axes([0.40, 0.05, 0.20, 0.03])

slider_thot  = Slider(ax_thot, 'T дна (°C)', 50.0, 100.0, valinit=T_hot_val, color='#fa5252')
slider_rad   = Slider(ax_rad, 'Радиус (мм)', 2.0, 10.0, valinit=R_mm_val, color='#4c6ef5')
slider_hconv = Slider(ax_hconv, 'Теплоотдача', 500.0, 4000.0, valinit=h_conv_val, color='#f59f00')

# Правая колонка: Текст результатов цикла
ax_text_cycle = fig.add_axes([0.70, 0.05, 0.25, 0.30])
ax_text_cycle.axis('off')
cycle_text = ax_text_cycle.text(0.0, 0.5, 'Ожидание завершения\nпервого цикла...', 
                                fontsize=11, va='center', linespacing=1.6, bbox=bbox_style)

# =============================================================================
# --- ЛОГИКА И ФИЗИКА ---
# =============================================================================

def update_params(val):
    global T_hot_val, R_mm_val, h_conv_val
    global t_hist, y_hist, T_hist, P_hist, V_hist, time_val
    
    T_hot_val = slider_thot.val
    R_mm_val = slider_rad.val
    h_conv_val = slider_hconv.val
    
    grad_img.set_data(np.linspace(T_hot_val, T_cold, 100).reshape(100, 1))
    
    # Очистка истории при смене параметров, чтобы график не ломался
    t_hist.clear()
    y_hist.clear()
    T_hist.clear()
    P_hist.clear()
    V_hist.clear()

slider_thot.on_changed(update_params)
slider_rad.on_changed(update_params)
slider_hconv.on_changed(update_params)

def update_physics():
    global y, v, T_drop, time_val
    global W_accum, Qin_accum, last_W, last_Qin, last_eff, state_in_air
    global current_P, current_V 

    T_water = T_hot_val - (T_hot_val - T_cold) * (y / H)
    rho_water = 998.0 - 0.4 * (T_water - 20.0)
    rho_aniline = 1022.0 - 1.2 * (T_drop - 20.0)

    R_base = R_mm_val / 1000.0  
    V_base = (4.0 / 3.0) * 3.14159 * R_base**3
    mass = 1022.0 * V_base  
    
    V_drop_m3 = mass / rho_aniline
    R_current = (0.75 * V_drop_m3 / 3.14159)**(1/3) 
    S_cross = 3.14159 * R_current**2      
    S_surf = 4.0 * 3.14159 * R_current**2
    
    C_p = 2000.0    
    h_conv = h_conv_val    
    
    dQ_dt = h_conv * S_surf * (T_water - T_drop)
    dT_drop = (dQ_dt / (mass * C_p)) * dt
    T_drop += dT_drop
    
    if dQ_dt > 0:
        Qin_accum += dQ_dt * dt  
        
    F_buoyancy = V_drop_m3 * 9.81 * rho_water
    F_gravity = mass * 9.81
    C_d = 0.47
    F_drag = -np.sign(v) * 0.5 * C_d * rho_water * S_cross * (v**2)
    
    F_net = F_buoyancy - F_gravity + F_drag
    a = F_net / mass
    
    v += a * dt  
    y += v * dt      
    
    if y > 0.005:  
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
    
    prev_P = current_P
    prev_V = current_V
    
    current_P = 101.325 + (rho_water * 9.81 * (H - y)) / 1000.0 
    current_V = V_drop_m3 * 1e6   
    
    dV_ml = current_V - prev_V
    dW_Joules = ((current_P + prev_P) / 2.0) * dV_ml * 1e-3
    W_accum += dW_Joules
    
    t_hist.append(time_val)
    y_hist.append(y * 100) 
    T_hist.append(T_drop)
    P_hist.append(current_P)
    V_hist.append(current_V)

    if len(t_hist) > max_history:
        t_hist.pop(0)
        y_hist.pop(0)
        T_hist.pop(0)
        P_hist.pop(0)
        V_hist.pop(0)

# =============================================================================
# --- АНИМАЦИЯ ---
# =============================================================================

def animate(frame):
    for _ in range(4):
        update_physics()
        
    drop_plot.set_offsets(np.c_[1, y])
    drop_plot.set_facecolor(cmap(norm(T_drop)))
    drop_plot.set_sizes([(R_mm_val / 5.0)**2 * 400])

    # Текст текущего состояния (в отдельной красивой плашке слева)
    temp_text.set_text(
        f"--- ТЕКУЩЕЕ СОСТОЯНИЕ ---\n"
        f"Температура среды: {T_hot_val - (T_hot_val - T_cold)*(y/H):.1f} °C\n"
        f"Температура капли: {T_drop:.1f} °C\n"
        f"Высота: {y * 100:.1f} см\n"
        f"Скорость: {v * 100:.1f} см/с\n"
        f"Объем капли: {current_V:.3f} мл"
    )
    
    line_yt.set_data(t_hist, y_hist)
    if time_val > 20: ax_yt.set_xlim(time_val - 20, time_val)
    else: ax_yt.set_xlim(0, 20)
        
    line_cycle.set_data(V_hist, P_hist)
    if len(V_hist) > 0:
        v_min, v_max = min(V_hist), max(V_hist)
        dv = (v_max - v_min) * 0.1 if v_max != v_min else 0.1
        ax_cycle.set_xlim(v_min - dv, v_max + dv)
        ax_cycle.set_ylim(101.0, 103.5)
    
    # Текст результатов цикла (в отдельной красивой плашке справа)
    cycle_text.set_text(
        f"--- РЕЗУЛЬТАТЫ ЦИКЛА ---\n"
        f"Подведено тепла (Q): {last_Qin:.2f} Дж\n"
        f"Совершена работа (A): {last_W * 1e6:.2f} мкДж\n"
        f"Тепловой КПД: {last_eff:.5f} %"
    )
    
    return drop_plot, temp_text, line_yt, line_cycle, cycle_text

ani = animation.FuncAnimation(fig, animate, interval=20, blit=False, cache_frame_data=False)

plt.show()