import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import Normalize
from matplotlib.widgets import Slider, RadioButtons

# =============================================================================
# --- ФИЗИЧЕСКИЕ ПАРАМЕТРЫ И СВОЙСТВА ВЕЩЕСТВ ---
# =============================================================================
H = 0.2            # Высота стакана (м)
T_cold = 20.0      # Температура на поверхности
dt = 0.05          # Шаг времени

# Свойства доступных веществ: базовая плотность (при 20°C), коэф. расширения, цветовая карта
substances = {
    'Анилин': {'rho0': 1022.0, 'beta': 1.2, 'cmap': 'coolwarm'},
    'Спец. масло': {'rho0': 1015.0, 'beta': 0.8, 'cmap': 'viridis'}
}

# Изменяемые параметры (стартовые значения)
T_hot_val = 80.0
R_mm_val = 5.0        
h_conv_val = 1500.0   
current_substance = 'Анилин'
graph_mode = 'P-V диаграмма'

# Переменные состояния
y, v, T_drop, time_val = 0.0, 0.0, 20.0, 0.0
max_history = 400
t_hist, y_hist, T_hist, P_hist, V_hist = [], [], [], [], []
current_P, current_V = 101.325, 0.0

W_accum, Qin_accum = 0.0, 0.0
last_W, last_Qin, last_eff = 0.0, 0.0, 0.0
state_in_air = False

# =============================================================================
# --- НАСТРОЙКА ИНТЕРФЕЙСА (ДАШБОРД) ---
# =============================================================================
fig = plt.figure(figsize=(15, 9))
fig.canvas.manager.set_window_title('Двигатель Дарлинга: Лабораторный стенд')
fig.patch.set_facecolor('#e9ecef') 

# --- ВЕРХНЯЯ ЗОНА: ГРАФИКИ ---
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
cmap = plt.get_cmap(substances[current_substance]['cmap'])

# 2. График Высоты
ax_yt = fig.add_axes([0.22, 0.45, 0.35, 0.45])
ax_yt.set_title('График высоты y(t)', fontweight='bold')
ax_yt.set_ylim(0, H * 100)
ax_yt.set_xlabel('Время, с')
ax_yt.grid(True, linestyle='--', alpha=0.7)
line_yt, = ax_yt.plot([], [], lw=2.5, color='#2b8a3e')

# 3. Вариативный график цикла (P-V или T-y)
ax_cycle = fig.add_axes([0.65, 0.45, 0.30, 0.45])
ax_cycle.set_title('Анализ цикла', fontweight='bold')
ax_cycle.grid(True, linestyle='--', alpha=0.7)
line_cycle, = ax_cycle.plot([], [], lw=2.5, color='#e67700')

# --- НИЖНЯЯ ЗОНА: ПАНЕЛЬ УПРАВЛЕНИЯ ---
bbox_style = dict(boxstyle='round,pad=1', facecolor='white', edgecolor='#ced4da', alpha=0.95)

# 1. Колонка 1: Текст состояния (Слева)
ax_text_state = fig.add_axes([0.05, 0.05, 0.20, 0.28])
ax_text_state.axis('off')
temp_text = ax_text_state.text(0.0, 0.5, '', fontsize=10, va='center', linespacing=1.5, bbox=bbox_style)

# 2. Колонка 2: Ползунки параметров (Центр-слева)
ax_thot  = fig.add_axes([0.30, 0.26, 0.20, 0.03])
ax_rad   = fig.add_axes([0.30, 0.16, 0.20, 0.03])
ax_hconv = fig.add_axes([0.30, 0.06, 0.20, 0.03])
slider_thot  = Slider(ax_thot, 'T дна (°C)', 50.0, 100.0, valinit=T_hot_val, color='#fa5252')
slider_rad   = Slider(ax_rad, 'Радиус (мм)', 2.0, 10.0, valinit=R_mm_val, color='#4c6ef5')
slider_hconv = Slider(ax_hconv, 'Теплоотдача', 500.0, 4000.0, valinit=h_conv_val, color='#f59f00')

# 3. Колонка 3: Переключатели (Центр-справа)
# Подняли верхний блок (y=0.23) и чуть сузили высоту
ax_radio_subst = fig.add_axes([0.55, 0.23, 0.15, 0.11])
ax_radio_subst.set_title('Вещество капли', fontsize=10, fontweight='bold')
radio_subst = RadioButtons(ax_radio_subst, tuple(substances.keys()))

# Опустили нижний блок (y=0.04), чтобы заголовок точно влез
ax_radio_graph = fig.add_axes([0.55, 0.04, 0.15, 0.11])
ax_radio_graph.set_title('Ось координат', fontsize=10, fontweight='bold')
radio_graph = RadioButtons(ax_radio_graph, ('P-V диаграмма', 'T-y диаграмма'))

# 4. Колонка 4: Текст результатов (Справа)
ax_text_cycle = fig.add_axes([0.75, 0.05, 0.20, 0.28])
ax_text_cycle.axis('off')
cycle_text = ax_text_cycle.text(0.0, 0.5, 'Ожидание цикла...', fontsize=10, va='center', linespacing=1.5, bbox=bbox_style)

# =============================================================================
# --- ЛОГИКА УПРАВЛЕНИЯ И ФИЗИКА ---
# =============================================================================

def clear_history():
    t_hist.clear(); y_hist.clear(); T_hist.clear(); P_hist.clear(); V_hist.clear()

def update_sliders(val):
    global T_hot_val, R_mm_val, h_conv_val
    T_hot_val, R_mm_val, h_conv_val = slider_thot.val, slider_rad.val, slider_hconv.val
    grad_img.set_data(np.linspace(T_hot_val, T_cold, 100).reshape(100, 1))
    clear_history()

def switch_substance(label):
    global current_substance, cmap
    current_substance = label
    cmap = plt.get_cmap(substances[label]['cmap']) # Меняем цвет капли
    clear_history()

def switch_graph(label):
    global graph_mode
    graph_mode = label
    clear_history()
    ax_cycle.clear() # Сбрасываем старые оси
    ax_cycle.set_title('Анализ цикла', fontweight='bold')
    ax_cycle.grid(True, linestyle='--', alpha=0.7)
    global line_cycle
    line_cycle, = ax_cycle.plot([], [], lw=2.5, color='#e67700')
    fig.canvas.draw_idle()

slider_thot.on_changed(update_sliders)
slider_rad.on_changed(update_sliders)
slider_hconv.on_changed(update_sliders)
radio_subst.on_clicked(switch_substance)
radio_graph.on_clicked(switch_graph)

def update_physics():
    global y, v, T_drop, time_val, W_accum, Qin_accum, last_W, last_Qin, last_eff, state_in_air
    global current_P, current_V 

    # Плотность среды
    T_water = T_hot_val - (T_hot_val - T_cold) * (y / H)
    rho_water = 998.0 - 0.4 * (T_water - 20.0)

    # Характеристики выбранного вещества
    subst_props = substances[current_substance]
    rho_drop = subst_props['rho0'] - subst_props['beta'] * (T_drop - 20.0)

    # Геометрия
    R_base = R_mm_val / 1000.0  
    V_base = (4.0 / 3.0) * np.pi * R_base**3
    mass = subst_props['rho0'] * V_base  # Масса фиксируется базовой плотностью
    
    V_drop_m3 = mass / rho_drop
    R_current = (0.75 * V_drop_m3 / np.pi)**(1/3) 
    S_cross = np.pi * R_current**2      
    S_surf = 4.0 * np.pi * R_current**2
    
    # Теплообмен
    dQ_dt = h_conv_val * S_surf * (T_water - T_drop)
    dT_drop = (dQ_dt / (mass * 2000.0)) * dt
    T_drop += dT_drop
    if dQ_dt > 0: Qin_accum += dQ_dt * dt  
        
    # Динамика
    F_buoyancy = V_drop_m3 * 9.81 * rho_water
    F_gravity = mass * 9.81
    F_drag = -np.sign(v) * 0.5 * 0.47 * rho_water * S_cross * (v**2)
    
    v += ((F_buoyancy - F_gravity + F_drag) / mass) * dt  
    y += v * dt      
    
    # Границы
    if y > 0.005: state_in_air = True 
    if y <= 0:
        y = 0.0; v = min(v, 0.0)
        if state_in_air:
            last_W, last_Qin = W_accum, Qin_accum
            last_eff = (abs(last_W) / abs(last_Qin)) * 100 if last_Qin != 0 else 0
            W_accum, Qin_accum, state_in_air = 0.0, 0.0, False
    elif y >= H:
        y = H; v = max(v, 0.0)
        
    time_val += dt
    
    # Термодинамика
    prev_P, prev_V = current_P, current_V
    current_P = 101.325 + (rho_water * 9.81 * (H - y)) / 1000.0 
    current_V = V_drop_m3 * 1e6   
    W_accum += ((current_P + prev_P) / 2.0) * (current_V - prev_V) * 1e-3
    
    # Запись истории
    t_hist.append(time_val); y_hist.append(y * 100); T_hist.append(T_drop)
    P_hist.append(current_P); V_hist.append(current_V)

    if len(t_hist) > max_history:
        t_hist.pop(0); y_hist.pop(0); T_hist.pop(0); P_hist.pop(0); V_hist.pop(0)

# =============================================================================
# --- АНИМАЦИЯ ---
# =============================================================================
def animate(frame):
    for _ in range(4): update_physics()
        
    drop_plot.set_offsets(np.c_[1, y])
    drop_plot.set_facecolor(cmap(norm(T_drop)))
    drop_plot.set_sizes([(R_mm_val / 5.0)**2 * 400])

    temp_text.set_text(
        f"--- СТАТУС {current_substance.upper()} ---\n"
        f"T воды: {T_hot_val - (T_hot_val - T_cold)*(y/H):.1f} °C\n"
        f"T капли: {T_drop:.1f} °C\n"
        f"Высота: {y * 100:.1f} см\n"
        f"Скорость: {v * 100:.1f} см/с\n"
        f"Объем: {current_V:.3f} мл"
    )
    
    line_yt.set_data(t_hist, y_hist)
    ax_yt.set_xlim(max(0, time_val - 20), max(20, time_val))
        
    # --- ЛОГИКА ПЕРЕКЛЮЧЕНИЯ ГРАФИКА ---
    if graph_mode == 'P-V диаграмма':
        line_cycle.set_data(V_hist, P_hist)
        ax_cycle.set_xlabel('Объем капли V, мл')
        ax_cycle.set_ylabel('Давление среды P, кПа')
        if len(V_hist) > 0:
            v_min, v_max = min(V_hist), max(V_hist)
            dv = (v_max - v_min) * 0.1 if v_max != v_min else 0.1
            ax_cycle.set_xlim(v_min - dv, v_max + dv)
            ax_cycle.set_ylim(101.0, 103.5)
    else:
        line_cycle.set_data(T_hist, y_hist)
        ax_cycle.set_xlabel('Температура капли T, °C')
        ax_cycle.set_ylabel('Высота y, см')
        if len(T_hist) > 0:
            ax_cycle.set_xlim(T_cold - 5, T_hot_val + 5)
            ax_cycle.set_ylim(-1, 21)
    
    cycle_text.set_text(
        f"--- ИТОГИ ЦИКЛА ---\n"
        f"Теплота (Q): {last_Qin:.2f} Дж\n"
        f"Работа (A): {last_W * 1e6:.2f} мкДж\n"
        f"КПД: {last_eff:.5f} %"
    )
    
    return drop_plot, temp_text, line_yt, line_cycle, cycle_text

ani = animation.FuncAnimation(fig, animate, interval=20, blit=False, cache_frame_data=False)
plt.show()