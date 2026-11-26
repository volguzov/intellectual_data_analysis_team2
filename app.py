"""Дашборды команды № 2. Запуск из корня репозитория: python app.py."""

import argparse

from flask import Flask, redirect
from taipy.gui import Gui

from overview import ALL, load_counts, month_choices, overview
from quality import BASE, load_predictions, quality


sections = [("overview", "Обзор"), ("quality", "Качество модели")]


def page(active, content):
    links = "  ".join(f"**{name}**" if path == active else f"[{name}](/{path})" for path, name in sections)
    return f"""
<|part|class_name=soc|
<|part|class_name=masthead|
<|part|class_name=brand|
SOC / Аналитика
|>
<|part|class_name=tabs|
{links}
|>
|>
{content}
|>
"""


PAGE = page("overview", """
# Поток карточек SOC

<|part|class_name=period|
<|{month}|selector|lov={periods}|dropdown|label=Месяц создания|on_change=change_period|>
<|{span}|text|class_name=context|>
|>

<|part|class_name=metrics|
<|part|class_name=kpi primary|
Карточек в выгрузке

<|{volume}|text|class_name=metric|>

За выбранный период
|>
<|part|class_name=kpi|
Без вердикта аналитика

<|{missing_rate}|text|class_name=metric|>

<|{missing_detail}|text|>
|>
<|part|class_name=kpi|
Максимум за день

<|{peak_value}|text|class_name=metric|>

<|{peak_date}|text|>
|>
|>

<|part|class_name=panel|
## Сколько карточек поступало в день?

<|chart|figure={flow_figure}|height=225px|width=100%|>
|>
<|part|class_name=comparisons|
<|part|class_name=panel|
## Какие типы преобладают?

Число карточек по типам · до шести самых частых

<|chart|figure={types_figure}|width=100%|>
|>
<|part|class_name=panel|
## Где чаще отсутствует вердикт?

Доля карточек без вердикта среди карточек каждого типа

<|chart|figure={rates_figure}|width=100%|>
|>
|>

<|part|class_name=source|
Использована учебная выгрузка SOC. Каждая карточка учтена один раз. Дата создания указана по московскому времени.
|>
""")

QUALITY_PAGE = page("quality", """
# Качество прогнозов

<|part|class_name=controls|
<|part|class_name=period|
<|{q_month}|selector|lov={q_months}|dropdown|label=Месяц оценки|on_change=change_quality|>
<|{q_span}|text|class_name=context|>
|>
<|part|class_name=threshold|
<|{q_threshold_label}|text|>
<|{q_threshold}|slider|min=0|max=100|step=1|continuous=False|on_change=change_quality|>
|>
|>
<|part|class_name=metrics|
<|part|class_name=kpi primary|
Пропущено инцидентов

<|{q_fn}|text|class_name=metric|>

<|{q_fn_detail}|text|>
|>
<|part|class_name=kpi|
Ложные тревоги

<|{q_false}|text|class_name=metric|>

Тревоги по легитимным событиям
|>
<|part|class_name=kpi|
Прогнозы с известным исходом

<|{q_total}|text|class_name=metric|>

<|{q_total_detail}|text|>
|>
|>

<|part|class_name=comparisons quality-grid|
<|part|class_name=panel|
## Где модель ошиблась?

Строки показывают вердикт аналитика, а столбцы показывают решение модели.

<|chart|figure={q_matrix}|height=265px|width=100%|>
|>
<|part|class_name=panel category-panel|
## В каких категориях больше пропусков?

Длина полосы показывает число пропусков. В подписи также указаны число подтверждённых инцидентов и доля пропусков.

<|chart|figure={q_groups}|width=100%|>
|>
<|part|class_name=panel|
## Как порог меняет число ошибок?

<|{q_scenario}|text|>

<|chart|figure={q_curve}|height=290px|width=100%|>
|>
|>
<|part|class_name=source|
<|{q_coverage}|text|>
|>
""")


def change_period(state, var_name, value):
    if value in overview_months:
        with state:
            for key, result in overview(counts, overview_months[value]).items():
                setattr(state, key, result)


def change_quality(state, var_name, value):
    month_value = value if var_name == "q_month" else state.q_month
    threshold_value = value if var_name == "q_threshold" else state.q_threshold
    with state:
        for key, result in quality(log, quality_months[month_value], threshold_value).items():
            setattr(state, key, result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Карточки SOC и качество прогнозов")
    parser.add_argument("--port", type=int, default=5062)
    params = parser.parse_args()
    try:
        counts = load_counts()
        log = load_predictions()
    except FileNotFoundError as error:
        parser.error(f"Файл {error.filename} не найден. Сохраните три Parquet из README.md в папку data/.")
    overview_months = month_choices(counts["day"].dt.strftime("%Y-%m"))
    periods = list(overview_months)
    month = ALL
    globals().update(overview(counts, month))
    quality_months = month_choices(log.month)
    q_months = list(quality_months)
    q_month, q_threshold = ALL, BASE
    globals().update(quality(log, q_month, q_threshold))
    server = Flask(__name__)
    server.add_url_rule("/", "home", lambda: redirect("/overview"))
    pages = {"overview": PAGE, "quality": QUALITY_PAGE}
    gui = Gui(pages=pages, flask=server, css_file="style.css")
    gui.run(host="127.0.0.1", port=params.port, title="Анализ карточек SOC",
            dark_mode=False, use_reloader=False, run_browser=False)
