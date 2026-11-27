"""Оценка решений на журнале задания 2 и карточках с вердиктом аналитика."""

from textwrap import wrap

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from overview import ALL, BLUE, DATA_DIR, TEAL, chart_style, date_label, number, percent


BASE = 18
ORANGE = "#ac491c"


def load_predictions():
    log = pd.read_parquet(DATA_DIR / "MODEL_PREDICTIONS_target.parquet", columns=[
        "incident_id", "timestamp", "true_incident_proba", "y_pred", "y"
    ]).rename(columns={"true_incident_proba": "score"})
    cards = pd.read_parquet(DATA_DIR / "iavd_incidents_raw3_target.parquet",
                            columns=["incident_id", "Пиктограмма IRP"])
    for frame in (cards, log):
        if frame.incident_id.isna().any() or not frame.incident_id.is_unique:
            raise ValueError("Идентификаторы должны быть заполнены и уникальны в каждой выгрузке.")
    if log.timestamp.isna().any() or not log.score.between(0, 1).all():
        raise ValueError("Некорректное время или оценка в журнале решений.")
    if not log.y_pred.eq(log.score.ge(BASE / 100)).all():
        raise ValueError("Исходный порог не воспроизводит решения журнала.")
    joined = log.merge(cards, how="left", on="incident_id", validate="one_to_one", indicator=True)
    joined["has_card"] = joined["_merge"].eq("both")
    joined["category"] = joined["Пиктограмма IRP"].fillna("Не указана").replace({
        "Атака на инфраструтуру": "Атака на инфраструктуру",
        "Управление УЗ": "Управление учётными записями",
    })
    joined["date"] = joined.timestamp.dt.tz_convert("Europe/Moscow")
    joined["month"] = joined.date.dt.strftime("%Y-%m")
    return joined


def error_matrix(tn, fp, fn, tp):
    counts = [[tn, fp], [fn, tp]]
    fig = go.Figure(go.Heatmap(
        z=[[0, 1], [1, 0]], x=["Тревоги нет", "Тревога"], y=["Легитимно", "Инцидент"],
        text=[[number(cell) for cell in row] for row in counts], texttemplate="%{text}",
        textfont=dict(size=20, color="#172e3a"), hovertemplate="Аналитик: %{y}<br>Модель: %{x}<br>Карточек: %{text}<extra></extra>",
        colorscale=[[0, "#e9eff5"], [1, "#f5dece"]], zmin=0, zmax=1,
        showscale=False, xgap=4, ygap=4,
    ))
    chart_style(fig, 235)
    fig.update_yaxes(autorange="reversed", title_text="Вердикт аналитика")
    fig.update_xaxes(title_text="Решение модели")
    return fig


def category_errors(positives):
    categories = positives.groupby("category").agg(
        actual=("alert", "size"), found=("alert", "sum"),
    )
    categories["missed"] = categories.actual - categories.found
    categories = categories.sort_values(["missed", "actual"], ascending=True)
    labels = [f"{number(r.missed)} из {number(r.actual)} · {percent(r.missed, r.actual)}"
              for r in categories.itertuples()]
    fig = go.Figure(go.Bar(
        y=categories.index.tolist(), x=categories.missed.tolist(), orientation="h",
        marker_color=ORANGE, width=0.48,
        hovertext=[f"{k}<br>Пропущено {number(r.missed)} из {number(r.actual)}"
                   f" ({percent(r.missed, r.actual)})" for k, r in categories.iterrows()],
        hovertemplate="%{hovertext}<extra></extra>",
    ))
    chart_style(fig, max(190, 60 * len(categories) + 62))
    # Keep the count and denominator visible even for a zero-length bar.
    ticks = ["<br>".join(wrap(name, 19)) + f"<br><b>{label}</b>"
             for name, label in zip(categories.index, labels)]
    fig.update_yaxes(type="category", tickmode="array", tickvals=categories.index.tolist(),
                     ticktext=ticks, tickfont=dict(size=12), showgrid=False)
    fig.update_xaxes(title_text="Число пропусков", rangemode="tozero",
                     tickformat=",~f", nticks=4,
                     range=[0, max(1, categories.missed.max() * 1.05)] if len(categories) else [0, 1])
    if categories.empty or categories.missed.max() == 0:
        finding = "В выбранном периоде пропусков нет."
    else:
        key = categories.missed.idxmax()
        row = categories.loc[key]
        finding = (f"Больше всего пропусков по числу карточек в категории «{key}»: "
                   f"{number(row.missed)} из {number(row.actual)} ({percent(row.missed, row.actual)}). "
                   f"По всем категориям доля пропусков составляет {percent(categories.missed.sum(), len(positives))}.")
    return fig, finding


def weekly_errors(dates, verified):
    days = dates.dt.tz_localize(None).dt.normalize()
    week = days.dt.to_period("W-SUN")
    periods = pd.period_range(week.min(), week.max(), freq="W-SUN")
    incidents = verified.loc[verified.y.eq(1)]
    weekly = pd.DataFrame({
        "forecasts": week.value_counts(),
        "known": verified.groupby(week).size(),
        "confirmed": incidents.groupby(week).size(),
        "missed": incidents.loc[~incidents.alert].groupby(week).size(),
    }).reindex(periods).fillna(0).astype(int)

    dates_shown, labels, rates, values = [], [], [], []
    first, last = days.min(), days.max()
    for row in weekly.itertuples():
        start = max(first, row.Index.start_time)
        end = min(last, row.Index.end_time)
        span = f"с {start:%d.%m} по {end:%d.%m}"
        dates_shown.append(span)
        if row.confirmed:
            detail = f"{number(row.missed)} из {number(row.confirmed)}"
            rates.append(100 * row.missed / row.confirmed)
            values.append(percent(row.missed, row.confirmed))
        else:
            detail = "0 подтверждённых" if row.known else "Нет известных исходов"
            if not row.forecasts:
                detail = "Нет прогнозов"
            rates.append(None)
            values.append("")
        labels.append(f"{span}<br>{detail}")

    fig = go.Figure(go.Bar(
        x=rates, y=dates_shown, orientation="h", marker_color=ORANGE, width=0.48,
        text=values, textposition="outside", cliponaxis=False, constraintext="none",
        customdata=weekly[["missed", "confirmed"]].values.tolist(),
        hovertemplate="%{y}<br>Пропущено %{customdata[0]} из %{customdata[1]} · %{text}<extra></extra>",
    ))
    chart_style(fig, 72 + 41 * len(weekly))
    fig.update_layout(margin_r=62, uniformtext_minsize=12, uniformtext_mode="show")
    fig.update_yaxes(type="category", tickvals=dates_shown, ticktext=labels,
                     autorange="reversed", showgrid=False)
    fig.update_xaxes(range=[0, 100], dtick=25, ticksuffix="%", side="top")
    comparable = weekly.loc[weekly.confirmed.gt(0)]
    if comparable.empty:
        finding = "В выбранном периоде нет подтверждённых инцидентов. Долю пропусков посчитать нельзя."
    else:
        fractions = comparable.missed / comparable.confirmed

        def describe(period):
            row = comparable.loc[period]
            span = dates_shown[weekly.index.get_loc(period)]
            return f"{span} ({number(row.missed)} из {number(row.confirmed)}, {percent(row.missed, row.confirmed)})"

        low, high = fractions.idxmin(), fractions.idxmax()
        if len(comparable) == 1:
            finding = f"Пропуски можно оценить только за неделю {describe(low)}."
        elif fractions.min() == fractions.max():
            finding = f"Во все недели с подтверждёнными инцидентами доля пропусков одинакова ({percent(comparable.missed.sum(), comparable.confirmed.sum())})."
        else:
            finding = (f"Самая высокая доля пропусков за неделю {describe(high)}. "
                       f"Самая низкая за неделю {describe(low)}.")
        finding += f" Недель с подтверждёнными инцидентами {len(comparable)} из {len(weekly)}."
    return fig, finding


def threshold_curve(data, selected):
    thresholds = np.arange(101)
    fig = go.Figure()
    for name, label, color in [(1, "Пропуски", ORANGE), (0, "Ложные тревоги", BLUE)]:
        scores = np.sort(data.loc[data.y.eq(name), "score"].to_numpy())
        below = np.searchsorted(scores, thresholds / 100, side="left")
        errors = below if name == 1 else len(scores) - below
        fig.add_scatter(x=thresholds.tolist(), y=errors.tolist(), mode="lines", name=label,
                        line=dict(color=color, dash="solid" if name == 1 else "dash"),
                        hovertemplate=f"{label}: %{{y}}<br>Порог: %{{x}}%<extra></extra>")
        fig.add_scatter(x=[selected], y=[int(errors[selected])], mode="markers",
                        marker=dict(color=color, size=9), showlegend=False, hoverinfo="skip")
    chart_style(fig, 290)
    fig.update_layout(showlegend=True, legend=dict(orientation="h", y=1.17))
    fig.update_xaxes(title_text="Порог тревоги", ticksuffix="%", range=[0, 100])
    fig.update_yaxes(title_text="Карточки", rangemode="tozero", gridcolor="#e4ebee", tickformat=",~f")
    return fig


def calibration(data):
    edges = [0, .1, .3, .5, .7, .9, float("inf")]
    bands = ["0 до 10%", "10 до 30%", "30 до 50%", "50 до 70%", "70 до 90%", "90 до 100%"]
    cut = pd.cut(data.score, edges, labels=bands, right=False)
    groups = data.groupby(cut, observed=True).agg(
        score=("score", "mean"), incidents=("y", "sum"), total=("y", "size")
    )
    actual = 100 * groups.incidents / groups.total
    details = [
        f"Оценки от {band}<br>Средняя оценка: {percent(row.score, 1)}"
        f"<br>Инциденты: {number(row.incidents)} из {number(row.total)} ({percent(row.incidents, row.total)})"
        for band, row in groups.iterrows()
    ]
    fig = go.Figure(go.Scatter(x=[0, 100], y=[0, 100], mode="lines",
                               line=dict(color="#9eafc1", dash="dot"), hoverinfo="skip"))
    featured = "50 до 70%"
    fig.add_scatter(x=(100 * groups.score).tolist(), y=actual.tolist(), mode="markers",
                    marker=dict(size=12, color=[ORANGE if s == featured else BLUE for s in groups.index]),
                    text=details, hovertemplate="%{text}<extra></extra>")
    chart_style(fig, 310)
    fig.update_xaxes(title_text="Средняя оценка модели", range=[-3, 103], dtick=25, ticksuffix="%")
    fig.update_yaxes(title_text="Доля инцидентов", range=[-3, 103], dtick=25, ticksuffix="%")
    if data.empty:
        finding = "За выбранный период нет прогнозов с известным исходом."
    elif featured in groups.index:
        row = groups.loc[featured]
        finding = (f"От 50% до 70%: средняя оценка {percent(row.score, 1)}. "
                   f"Подтверждено {number(row.incidents)} из {number(row.total)} ({percent(row.incidents, row.total)}).")
    else:
        finding = "В диапазоне от 50% до 70% нет прогнозов с известным исходом."
    return fig, finding


def quality(log, month, threshold):
    threshold = int(threshold)
    if not 0 <= threshold <= 100:
        raise ValueError("Порог должен быть от 0 до 100 процентов.")
    cohort = log if month == ALL else log.loc[log.month.eq(month)]
    if cohort.empty:
        raise ValueError("В выбранном периоде отсутствуют прогнозы.")
    eligible = cohort.has_card & cohort.y.isin([0, 1])
    data = cohort.loc[eligible].copy()
    data["alert"] = data.score.ge(threshold / 100)
    pos = data.y.eq(1)
    tp = int((pos & data.alert).sum())
    fn = int(pos.sum()) - tp
    fp = int((~pos & data.alert).sum())
    tn = len(data) - tp - fn - fp
    base_fn = int((pos & data.y_pred.eq(0)).sum())
    base_fp = int((~pos & data.y_pred.eq(1)).sum())
    fig, finding = category_errors(data.loc[pos])
    weekly_fig, weekly_text = weekly_errors(cohort.date, data)
    calibration_fig, calibration_text = calibration(data)
    positives = tp + fn
    negatives = tn + fp
    comparison = pd.DataFrame({
        "Показатель": ["Верные решения", "Обнаружено инцидентов", "Пропущено инцидентов", "Ложные тревоги"],
        "Модель": [percent(tp + tn, len(data)) if len(data) else "Нет данных", number(tp), number(fn), number(fp)],
        "Всё легитимно": [percent(negatives, len(data)) if len(data) else "Нет данных",
                          "0", number(positives), "0"],
    })
    missed = f"{number(fn)} из {number(fn + tp)} · {percent(fn, fn + tp)}" if fn + tp else "Нет подтверждённых инцидентов"
    workload = int(cohort.score.ge(threshold / 100).sum())
    original_workload = int(cohort.y_pred.eq(1).sum())
    scenario = (f"При пороге {threshold}% по всему журналу выбранного периода получается "
                f"{number(workload)} тревог, из них {number(workload - tp - fp)} с неизвестным исходом. "
                f"По проверяемым карточкам пропущено {number(fn)} инцидентов, ложных тревог {number(fp)}.")
    if threshold != BASE:
        scenario += (f" При 18% смена получила {number(original_workload)} тревог. "
                     f"По проверяемым карточкам было {number(base_fn)} пропусков и {number(base_fp)} ложных тревог.")
    return {
        "q_threshold_label": f"Порог тревоги · {threshold}%",
        "q_fn": number(fn),
        "q_fn_detail": (f"Из {number(fn + tp)} подтверждённых · {percent(fn, fn + tp)}"
                        if fn + tp else "Нет подтверждённых инцидентов"),
        "q_total": number(len(data)),
        "q_total_detail": f"Из {number(len(cohort))} в журнале",
        "q_span": f"С {date_label(cohort.date.min())} по {date_label(cohort.date.max())}",
        "q_coverage": f"Исход известен у {number(len(data))} из {number(len(cohort))} прогнозов. "
                      f"Совпадений по номеру карточки нет у {number((~cohort.has_card).sum())}. "
                      f"Ещё для {number((cohort.has_card & ~eligible).sum())} исход не определён.",
        "q_missed": missed, "q_false": number(fp),
        "q_matrix": error_matrix(tn, fp, fn, tp), "q_groups": fig, "q_group_text": finding,
        "q_curve": threshold_curve(data, threshold), "q_weekly": weekly_fig, "q_weekly_text": weekly_text,
        "q_calibration": calibration_fig, "q_calibration_text": calibration_text,
        "q_comparison": comparison,
        "q_scenario": scenario,
    }
