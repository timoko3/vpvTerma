import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# --- Исходные параметры ---
n_moles = 1.0       # Количество вещества (моль)
R = 8.314           # Универсальная газовая постоянная (Дж/(моль*К))
gamma = 1.4         # Показатель адиабаты (для двухатомного газа)

# --- Настройка окна графика ---
fig, ax = plt.subplots(figsize=(10, 7))
plt.subplots_adjust(bottom=0.35) # Оставляем место снизу для ползунков
ax.set_title('Интерактивный цикл Карно (P-V диаграмма)')
ax.set_xlabel('Объем V, м³')
ax.set_ylabel('Давление P, Па')
ax.grid(True, linestyle='--', alpha=0.6)

# --- Создание осей для ползунков ---
ax_TH = plt.axes([0.15, 0.20, 0.75, 0.03])
ax_TC = plt.axes([0.15, 0.15, 0.75, 0.03])
ax_V1 = plt.axes([0.15, 0.10, 0.75, 0.03])
ax_V2 = plt.axes([0.15, 0.05, 0.75, 0.03])

# --- Инициализация ползунков ---
slider_TH = Slider(ax_TH, 'T_нагр (K)', 300, 1000, valinit=600, valstep=10)
slider_TC = Slider(ax_TC, 'T_холод (K)', 100, 500, valinit=300, valstep=10)
slider_V1 = Slider(ax_V1, 'V1 (м³)', 0.01, 0.05, valinit=0.02, valstep=0.001)
slider_V2 = Slider(ax_V2, 'V2 (м³)', 0.051, 0.15, valinit=0.06, valstep=0.001)

# --- Пустые объекты для линий и текста ---
line_iso_exp, = ax.plot([], [], 'r-', linewidth=2, label='Изотерма (нагрев)')
line_adi_exp, = ax.plot([], [], 'g-', linewidth=2, label='Адиабата (расширение)')
line_iso_comp, = ax.plot([], [], 'b-', linewidth=2, label='Изотерма (охлаждение)')
line_adi_comp, = ax.plot([], [], 'm-', linewidth=2, label='Адиабата (сжатие)')
fill_area = None

text_eff = ax.text(0.65, 0.95, '', transform=ax.transAxes, fontsize=12,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

# --- Функция обновления графика при движении ползунков ---
def update(val):
    global fill_area
    TH = slider_TH.val
    TC = slider_TC.val
    V1 = slider_V1.val
    V2 = slider_V2.val

    # Защита от физически некорректных значений
    if TC >= TH:
        slider_TC.set_val(TH - 10)
        return
    if V1 >= V2:
        slider_V1.set_val(V2 - 0.01)
        return

    # Расчет узловых точек цикла
    # Точка 1
    P1 = n_moles * R * TH / V1
    # Точка 2
    P2 = n_moles * R * TH / V2

    # Точка 3 (после адиабатического расширения)
    # Формула: T1*V1^(gamma-1) = T2*V2^(gamma-1)
    V3 = V2 * (TH / TC)**(1 / (gamma - 1))
    P3 = n_moles * R * TC / V3

    # Точка 4 (после адиабатического сжатия к начальной изотерме)
    V4 = V1 * (TH / TC)**(1 / (gamma - 1))
    P4 = n_moles * R * TC / V4

    # Генерация массивов для отрисовки кривых
    V_iso1 = np.linspace(V1, V2, 100)
    P_iso1 = n_moles * R * TH / V_iso1

    V_adi1 = np.linspace(V2, V3, 100)
    P_adi1 = P2 * (V2 / V_adi1)**gamma

    V_iso2 = np.linspace(V3, V4, 100)
    P_iso2 = n_moles * R * TC / V_iso2

    V_adi2 = np.linspace(V4, V1, 100)
    P_adi2 = P4 * (V4 / V_adi2)**gamma

    # Обновление данных линий
    line_iso_exp.set_data(V_iso1, P_iso1)
    line_adi_exp.set_data(V_adi1, P_adi1)
    line_iso_comp.set_data(V_iso2, P_iso2)
    line_adi_comp.set_data(V_adi2, P_adi2)

    # Обновление заштрихованной площади (полезная работа)
    if fill_area is not None:
        fill_area.remove()
    
    V_full = np.concatenate([V_iso1, V_adi1, V_iso2, V_adi2])
    P_full = np.concatenate([P_iso1, P_adi1, P_iso2, P_adi2])
    fill_area = ax.fill(V_full, P_full, alpha=0.3, color='orange')[0]

    # Расчет термодинамических показателей
    eff = 1 - (TC / TH)
    Q_in = n_moles * R * TH * np.log(V2 / V1)
    Work = Q_in * eff

    # Обновление текста
    text_eff.set_text(
        f'КПД: {eff*100:.1f}%\n'
        f'Подведенная теплота: {Q_in:.0f} Дж\n'
        f'Полезная работа: {Work:.0f} Дж'
    )

    # Динамическое масштабирование осей
    ax.set_xlim(min(V1, V4)*0.8, max(V2, V3)*1.1)
    ax.set_ylim(min(P3, P4)*0.8, max(P1, P2)*1.1)
    fig.canvas.draw_idle()

# --- Привязка функции обновления к ползункам ---
slider_TH.on_changed(update)
slider_TC.on_changed(update)
slider_V1.on_changed(update)
slider_V2.on_changed(update)

# Первичная отрисовка
update(0)
ax.legend(loc='upper right')

# Запуск интерактивного окна
plt.show()