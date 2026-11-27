"""Дашборды команды № 2. Запуск из корня репозитория: python app.py."""

import argparse

include_shap = False
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Карточки SOC и прогнозы модели")
    parser.add_argument("--port", type=int, default=5062)
    parser.add_argument("--shap", action="store_true", help="Открыть задания 1, 2 и 3")
    params = parser.parse_args()
    include_shap = params.shap
    if include_shap:
        # На Windows LightGBM нужно загрузить до PyArrow.
        try:
            from explanations import Explanations
            from source_removal import external_result_table
        except ModuleNotFoundError as error:
            parser.error(f"Не найден модуль {error.name}. Проверьте распаковку комплекта курса и установку requirements_shap.txt по README.")

from flask import Flask, redirect
from taipy.gui import Gui, invoke_long_callback

from overview import ALL, handling_summary, load_counts, load_handling, month_choices, overview
from quality import BASE, load_predictions, quality


sections = [("overview", "Обзор"), ("quality", "Качество модели")]
if include_shap:
    sections.append(("explain", "Почему так?"))


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

Шесть самых частых типов за выбранный период.

<|chart|figure={types_figure}|width=100%|>

<|{kinds_text}|text|>
|>
<|part|class_name=panel|
## Сколько времени занимает разбор карточки?

<|{time_summary}|text|>

<|chart|figure={time_chart}|width=100%|>

Интервал от взятия в работу до закрытия включает паузы. Нижняя граница каждого интервала включена.

<|{time_coverage}|text|>
|>
|>
<|part|class_name=comparisons|
<|part|class_name=panel|
## Где чаще отсутствует вердикт?

Сравниваем доли без вердикта внутри шести самых частых типов.

<|chart|figure={rates_figure}|width=100%|>

<|{rates_text}|text|>

<|{scan_text}|text|>
|>
<|part|class_name=panel|
## Какие правила дают больше всего ложных тревог?

<|{noise_summary}|text|>

<|chart|figure={rule_chart}|width=100%|>

По длине полосы видно, с какого правила начать проверку настроек. Карточка с несколькими правилами учитывается один раз у каждого из них.
|>
|>

<|part|class_name=source|
Период определяется датой создания карточки по московскому времени.
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

<|part|class_name=comparisons|
<|part|class_name=panel|
## Сколько инцидентов обнаруживает модель?

Строки показывают вердикт аналитика, а столбцы показывают решение модели.

<|chart|figure={q_matrix}|height=265px|width=100%|>

Если считать все карточки легитимными, каждое легитимное событие даст верный ответ, а каждый инцидент будет пропущен. Сравним оба способа на тех же карточках.

<|{q_comparison}|table|show_all|sortable=False|filter=False|>
|>
<|part|class_name=panel|
## В каких категориях больше пропусков?

Длина полосы показывает число пропусков. В подписи также указаны число подтверждённых инцидентов и доля пропусков.

<|chart|figure={q_groups}|width=100%|>

<|{q_group_text}|text|>
|>
|>
<|part|class_name=panel|
## Как менялась доля пропущенных инцидентов по неделям?

Под датами указано, сколько инцидентов модель пропустила из числа подтверждённых. Полоса показывает долю пропусков при выбранном пороге. Недели считаем с понедельника по московскому времени. Если в неделе нет подтверждённых инцидентов, доля пропусков для неё не определена.

<|chart|figure={q_weekly}|width=100%|>

<|{q_weekly_text}|text|>
|>
<|part|class_name=comparisons|
<|part|class_name=panel|
## Как оценка модели соотносится с долей инцидентов?

Перед выбором порога сравним оценки с исходами. Каждая точка соответствует одному диапазону оценок. На пунктире средняя оценка совпадает с долей инцидентов.

<|chart|figure={q_calibration}|width=100%|>

<|{q_calibration_text}|text|>
|>
<|part|class_name=panel|
## Как порог меняет нагрузку на смену?

Порог определяет, по каким карточкам смена получит тревогу. Обе линии рассчитаны на карточках с известным исходом.

<|chart|figure={q_curve}|height=290px|width=100%|>

<|{q_scenario}|text|>
|>
|>
<|part|class_name=source|
<|{q_coverage}|text|>
|>
""")


EXPLAIN_PAGE = page("explain", """
# Почему модель так решила?

Сравните пропуск № 523952 с ложной тревогой № 527002. В наборе можно выбрать любую из восьми карточек. Здесь используется отдельная учебная модель со своим порогом.

<|part|class_name=period|
<|{x_selected}|selector|lov={x_options}|dropdown|label=Карточка и исход|on_change=select_card|>
<|Набор № 2 · LightGBM v001|text|class_name=context|>
|>
<|{x_status}|text|class_name=status|>
<|part|render={x_ready}|
<|part|class_name=metrics|
<|part|class_name=kpi primary|
Исход проверки

<|{x_outcome}|text|class_name=metric words|>

<|{x_verdict}|text|>
|>
<|part|class_name=kpi|
Оценка модели

<|{x_probability}|text|class_name=metric|>

<|{x_decision}|text|>
|>
<|part|class_name=kpi|
Порог тревоги

<|{x_threshold}|text|class_name=metric|>

Порог учебной модели
|>
|>

<|part|class_name=panel|
## Что повлияло на решение?

Каждая полоса показывает изменение балла: вправо он растёт, влево снижается. Пунктир обозначает порог тревоги, точка показывает итог.

<|chart|figure={x_water}|height=548px|width=100%|>

<|{x_note}|text|>

<|Как проверить расчёт по исходным полям|expandable|expanded=False|
<|{x_address_check}|text|>

<|{x_check}|text|>
<|chart|figure={x_top}|height=400px|width=100%|>
<|{x_table}|table|show_all|>
|>
|>

<|part|class_name=comparisons|
<|part|class_name=panel|
## Какие поля важны для модели?

Пять полей с наибольшим средним вкладом по модулю. Среди 320 карточек поровну примеров каждого исхода. При другом составе выборки порядок может измениться.

<|chart|figure={x_global}|height=250px|width=100%|>

В этой выборке первые три места занимают имя хоста источника, имя хоста цели и доля частных IP источника.
|>
<|part|class_name=panel|
## Как меняется оценка с длиной алерта?

У карточек 523952 и 527002 длина сообщения дала вклады разных знаков. Чтобы оценить её влияние в среднем, на 3 173 записях меняем только признак длины. Точки с одинаковой длиной объединены.

<|chart|figure={x_pdp}|height=250px|width=100%|>

<|{x_pdp_summary}|text|>
|>
|>

<|part|class_name=panel|
## Как часто модель ошибается на карточках с записанным внешним адресом?

В исходной выгрузке 259 218 уникальных карточек. В 42 052 из них в src записан хотя бы один публичный адрес IPv4 или IPv6. Групповые и зарезервированные адреса исключили. Сюда входят 285 карточек со смешанным списком адресов. У 15 835 карточек с внешним src аналитик поставил «Решён» или «False Positive».

Ложные тревоги в таблице посчитаны с исходным адресом. Для подсчёта смен решений удалили всё поле src и повторили прогноз при пороге 1,9259%. Период после обучения начинается 1 сентября 2025 года.

<|{x_external_results}|table|show_all|sortable=False|filter=False|>

После обучения исчезли 5 прежних ложных тревог и появились 4 новые. Модель пропустила 16 ранее обнаруженных инцидентов и нашла 2 прежних пропуска. В итоге без src она дала на одну ложную тревогу меньше и обнаружила на 14 инцидентов меньше. Для этой модели сохраняем записанный адрес.
|>

<|part|render={x_show_source_test}|
<|Опыты с пустым адресом в учебной выборке|expandable|expanded=False|
### Изменится ли решение, если указать адрес источника?

В двух копиях карточки № 523952 изменили только поле src. Для каждой копии заново рассчитали признаки и прогноз при том же пороге. В таблице оценки округлены до шести знаков после запятой.

<|{x_source_scenarios}|table|show_all|sortable=False|filter=False|>

### Сколько решений изменится на всей выборке?

В выборке из 320 карточек отобрали все карточки с пустым src. Каждую проверили с теми же двумя адресами при прежнем пороге. Новую тревогу сверили с вердиктом аналитика, чтобы отличить обнаруженный инцидент от ложной тревоги.

<|{x_source_transitions}|table|show_all|sortable=False|filter=False|>

В учебную выборку отобрали по 80 карточек каждого исхода. Эти числа описывают чувствительность модели к двум подстановкам в этой выборке.

<|part|class_name=source-crossings|
### Насколько близко к порогу были 13 новых тревог?

Исходные оценки семи подтверждённых инцидентов и шести ложных тревог. Все 13 карточек пересекли порог после подстановки 1.1.1.1.

<|chart|figure={x_crossing_chart}|height=455px|width=100%|>
|>

<|{x_crossing_summary}|text|>

Семь обнаруженных инцидентов относятся к семи разным правилам. Среди шести ложных тревог четыре пришлись на два правила. Одно из них встречается также у подтверждённого инцидента.

<|Оценки и правила по каждой карточке|expandable|expanded=False|
Разрыв до порога указан в процентных пунктах. В последнем столбце приведено правило, записанное в карточке.

<|{x_crossing_cards}|table|show_all|sortable=False|filter=False|>
|>

<|Три карточки одного правила о вредоносном ПО|expandable|expanded=False|
### Почему одно правило дало верную и две ложные тревоги?

Правило TTS_10.1_Malware_Found_And_Not_Deleted встретилось в трёх карточках, где после подстановки 1.1.1.1 появилась тревога. У № 522033 аналитик указал четыре вредоносных файла в кэше браузера и подтвердил их удаление. В № 527995 тоже фигурировал кэш браузера, но проверка хэша по источникам сведений об угрозах не дала подтверждения. Событие закрыли как ложную тревогу. № 526902 аналитик описал как повторяющееся шумное срабатывание. В выгрузке у № 522033 указан Kaspersky и заполнено поле «Алерт». У двух карточек с вердиктом False Positive оба поля пусты.

При пороге 1,9259% оценка инцидента № 522033 выросла с 0,8493% до 14,9296%. У ложных тревог оценка № 527995 выросла с 0,6451% до 5,7365%, а № 526902 с 0,0884% до 2,3357%.

Таблица показывает три крупнейших положительных изменения вкладов SHAP у каждой карточки. Остальные 70 полей объединены в одной строке. Сумма изменений даёт прирост балла в логарифмах шансов.

<|{x_rule_shap}|table|show_all|sortable=False|filter=False|>

Во всех трёх случаях доля частных IP источника до подстановки была неизвестна. При подготовке данных модель подставляла обучающую медиану 1,0. С адресом 1.1.1.1 доля стала равна 0. Изменились поле src и три вычисленных признака. Модель пересчитала число элементов в списке src, число распознанных IP и долю частных адресов. В деревьях вклад одного поля зависит от значений других, поэтому SHAP изменился и у полей, значения которых остались прежними.

### Хватило ли изменения одной доли частных IP для перехода через порог?

В исходных карточках поле src пустое, и модель подставляет в признак src_is_private медиану 1. Перед масштабированием задали только этому признаку значение 0. Поле src осталось пустым, число адресов осталось равным нулю. Оценки получили повторным прогнозом модели при прежнем пороге 1,9259%.

<|{x_private_share}|table|show_all|sortable=False|filter=False|>

Одной доли хватило для обнаружения инцидента № 522033 и для ложной тревоги № 527995. У № 526902 оценка осталась на 0,4995 процентного пункта ниже порога. Эта карточка перешла порог при полной подстановке адреса, когда вместе с долей изменились и другие входы модели.

### Что перевело карточку № 526902 через порог?

Долю частных IP оставили равной 0 во всех восьми вариантах. Меняли три входа модели после вычисления признаков. В столбцах меняется значение поля src, в строках меняются число элементов в списке src (len_src) и число распознанных IP (ip_count_src). Поэтому отдельные сочетания, например пустое поле и один распознанный адрес, возможны только в этом эксперименте. Каждую оценку сравнивали с порогом 1,9259% до округления.

<|{x_address_combinations}|table|show_all|sortable=False|filter=False|>

При обоих нулевых счётчиках замена пустого src на 1.1.1.1 повысила оценку с 1,4264% до 2,1938%. Это на 0,2679 процентного пункта выше порога. С пустым src даже оба счётчика, равные 1, дали лишь 1,5195%. Для этой карточки при доле частных IP, равной 0, достаточно одного дополнительного изменения поля src. Во всех четырёх вариантах с адресом модель подала ложную тревогу.

### Возникает ли тревога с другими внешними адресами?

Для № 526902 зафиксировали долю частных IP, число элементов в src и число распознанных IP на нуле. Меняли только текстовое значение src. Сравнили пустое поле и шесть заранее выбранных внешних IPv4, включая 1.1.1.1. Порог модели равен 1,9259%.

<|{x_external_addresses}|table|show_all|sortable=False|filter=False|>

Из пяти дополнительных адресов тревогу вызвал только 1.1.1.2. Для четырёх остальных уже при кодировании src методом TF‑IDF получился тот же числовой вектор, что и для пустого поля. Совпадение оценок сохраняется до округления. На этой карточке реакция зависит от конкретного адреса. Частоту ложных тревог в журнале по таким подстановкам оценить нельзя.
|>
|>
|>
|>
""")


def apply_card(state, success, label, result=None):
    if state.x_selected != label:
        return
    with state:
        if success:
            for name, value in result.items():
                setattr(state, name, value)
        state.x_ready = success
        state.x_status = "" if success else "Расчёт не прошёл проверку. Повторно выберите карточку."


def select_card(state, var_name, value):
    if value in x_options:
        with state:
            state.x_ready = False
            state.x_status = "Рассчитываем оценку и вклады признаков…"
        invoke_long_callback(state, explanations.explain, (value,), apply_card, (value,))


def overview_page(month):
    return {**overview(counts, month), **handling_summary(handling_cards, month)}


def change_period(state, var_name, value):
    if value in overview_months:
        with state:
            for key, result in overview_page(overview_months[value]).items():
                setattr(state, key, result)


def change_quality(state, var_name, value):
    month_value = value if var_name == "q_month" else state.q_month
    threshold_value = value if var_name == "q_threshold" else state.q_threshold
    with state:
        for key, result in quality(log, quality_months[month_value], threshold_value).items():
            setattr(state, key, result)


if __name__ == "__main__":
    try:
        counts = load_counts()
        handling_cards = load_handling()
        log = load_predictions()
    except FileNotFoundError as error:
        parser.error(f"Файл {error.filename} не найден. Сохраните три Parquet из README.md в папку data/.")
    overview_months = month_choices(counts["day"].dt.strftime("%Y-%m"))
    periods = list(overview_months)
    month = ALL
    globals().update(overview_page(month))
    quality_months = month_choices(log.month)
    q_months = list(quality_months)
    q_month, q_threshold = ALL, BASE
    globals().update(quality(log, q_month, q_threshold))
    server = Flask(__name__)
    server.add_url_rule("/", "home", lambda: redirect("/overview"))
    pages = {"overview": PAGE, "quality": QUALITY_PAGE}
    if include_shap:
        try:
            explanations = Explanations()
        except FileNotFoundError as error:
            parser.error(f"Не найден файл {error.filename}. Проверьте оба комплекта задания 3 по README.")
        x_options = list(explanations.options)
        x_selected = x_options[0]
        x_global, x_pdp = explanations.global_fig, explanations.pdp_fig
        x_pdp_summary = explanations.pdp_summary
        x_source_scenarios = explanations.source_address_scenarios()
        x_source_transitions = explanations.source_address_transitions()
        x_crossing_chart, x_crossing_cards, x_crossing_summary = explanations.external_address_crossings()
        x_rule_shap = explanations.malware_rule_shap(x_crossing_cards["Карточка"].astype(int))
        x_private_share = explanations.private_share_scenarios([522033, 527995, 526902])
        x_address_combinations = explanations.address_combinations_526902()
        x_external_addresses = explanations.external_addresses_526902()
        x_external_results = external_result_table()
        x_ready, x_status = True, ""
        globals().update(explanations.explain(x_selected))
        pages["explain"] = EXPLAIN_PAGE
    gui = Gui(pages=pages, flask=server, css_file="style.css")
    gui.run(host="127.0.0.1", port=params.port, title="Анализ карточек SOC",
            dark_mode=False, use_reloader=False, run_browser=False)
