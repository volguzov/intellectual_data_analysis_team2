"""Обзор карточек SOC из задания 1. Каждая карточка выгрузки учитывается один раз."""

from pathlib import Path
from textwrap import wrap

import pandas as pd
import plotly.graph_objects as go


DATA_DIR = Path(__file__).resolve().parent / "data"
DATA = DATA_DIR / "iavd_incidents_raw_target.parquet"
ALL = "Весь период"
BLUE = "#315d90"
TEAL = "#4b799b"
MONTHS = "января февраля марта апреля мая июня июля августа сентября октября ноября декабря".split()
MONTH_NAMES = "Январь Февраль Март Апрель Май Июнь Июль Август Сентябрь Октябрь Ноябрь Декабрь".split()


def number(n):
    return f"{n:,.0f}".replace(",", " ")


def percent(n, total):
    return f"{100 * n / total:.1f}".replace(".", ",") + "%"


def date_label(day):
    return f"{day.day} {MONTHS[day.month - 1]} {day.year}"


def month_choices(months):
    choices = {ALL: ALL}
    for month in sorted(set(months), reverse=True):
        year, month_number = month.split("-")
        choices[f"{MONTH_NAMES[int(month_number) - 1]} {year}"] = month
    return choices


def load_counts(path=DATA):
    cards = pd.read_parquet(path, columns=["EntityId", "Создан", "Тип инцидента", "Вердикт"])
    if cards["EntityId"].isna().any() or not cards["EntityId"].is_unique:
        raise ValueError("Для обзора требуется ровно одна строка на идентификатор карточки.")
    created = cards["Создан"]
    if created.isna().any() or created.dt.tz is None:
        raise ValueError("Для каждой карточки требуется дата создания с часовым поясом.")
    day = created.dt.tz_convert("Europe/Moscow").dt.tz_localize(None).dt.normalize()
    kind = cards["Тип инцидента"].fillna("").str.strip().replace("", "Тип не указан")
    kind = kind.mask(kind.str.contains("~~||~~", regex=False), "Несколько типов")
    verdict = cards["Вердикт"].fillna("").str.strip()
    rows = pd.DataFrame({"day": day, "kind": kind, "missing": verdict.isin(["", "Не указан"])})
    return rows.groupby(["day", "kind"], as_index=False).agg(
        cards=("missing", "size"), missing=("missing", "sum")
    )


def chart_style(fig, height):
    fig.update_layout(
        template="plotly_white", height=height, showlegend=False,
        font=dict(family="Segoe UI, Arial, sans-serif", size=13, color="#334c65"),
        colorway=[BLUE, "#ab542e", TEAL],
        paper_bgcolor="white", plot_bgcolor="white",
        margin=dict(l=10, r=22, t=12, b=36),
        separators=", ", hoverlabel=dict(align="left"),
    )
    fig.update_xaxes(automargin=True, zeroline=False, gridcolor="#e5ebf1")
    fig.update_yaxes(automargin=True, zeroline=False, gridcolor="#e5ebf1")
    return fig


def load_handling(path=DATA):
    cards = pd.read_parquet(path, columns=[
        "EntityId", "Создан", "Вердикт", "Правило корреляции SIEM",
        "Дата взятия в работу", "Дата закрытия (факт)",
    ]).set_index("EntityId", verify_integrity=True)
    created = cards["Создан"]
    taken = cards["Дата взятия в работу"]
    closed = cards["Дата закрытия (факт)"]
    both = taken.notna() & closed.notna()
    ordered = taken.ge(created) & closed.ge(taken)
    verdict = cards["Вердикт"].fillna("").str.strip()
    return pd.DataFrame({
        "month": created.dt.tz_convert("Europe/Moscow").dt.strftime("%Y-%m"),
        "fp": verdict.eq("False Positive"),
        "missing_verdict": verdict.isin(["", "Не указан"]),
        "rule": cards["Правило корреляции SIEM"],
        "dates_missing": ~both,
        "minutes": (closed - taken).dt.total_seconds().div(60).where(ordered),
    })


def handling_bars(counts, ticks, axis, height):
    fig = go.Figure(go.Bar(
        x=counts.tolist(), y=counts.index.tolist(), orientation="h", marker_color=BLUE,
        text=[number(n) for n in counts], textposition="outside", cliponaxis=False,
        hovertemplate="%{y}<br>Карточек: %{text}<extra></extra>",
    ))
    chart_style(fig, height)
    fig.update_yaxes(type="category", tickvals=counts.index.tolist(), ticktext=ticks,
                     autorange="reversed", showgrid=False)
    fig.update_xaxes(title_text=axis, range=[0, 1.25 * max(1, counts.max())] if len(counts) else [0, 1],
                     tickformat=",~f", nticks=4)
    return fig


def handling_summary(cards, month):
    part = cards if month == ALL else cards.loc[cards.month.eq(month)]
    false_alarms = part.loc[part.fp, "rule"]
    names = false_alarms.str.split("~~||~~", regex=False).explode().str.strip().dropna()
    pairs = names[names.ne("")].reset_index().drop_duplicates(["EntityId", "rule"])
    ranked = pairs.groupby("rule", sort=True).size().sort_values(ascending=False, kind="stable").head(5)
    missing_rule = len(false_alarms) - pairs.EntityId.nunique()
    rule_ticks = [
        "<br>".join(line.replace("_ ", "_") for line in wrap(name.replace("_", "_ "), 23))
        for name in ranked.index
    ]
    lines = max((tick.count("<br>") + 1 for tick in rule_ticks), default=1)
    rules_fig = handling_bars(ranked, rule_ticks, "Ложные тревоги",
                              max(175, 65 + len(ranked) * (16 * lines + 9)))
    if ranked.empty:
        rules_fig.add_annotation(x=.5, y=.5, xref="paper", yref="paper", showarrow=False,
                                 text="Нет ложных тревог с указанным правилом")
    noise = (f"Вердикт False Positive у {number(len(false_alarms))} из {number(len(part))} карточек "
             f"({percent(len(false_alarms), len(part))}). Без названия правила: {number(missing_rule)}.")
    scan = part.loc[part.rule.eq("GL_FW_S04_Host_Port_Scan")]
    scan_text = (f"Правило GL_FW_S04_Host_Port_Scan указано единственным в {number(len(scan))} карточках. "
                 f"Без вердикта осталось {number(scan.missing_verdict.sum())}, "
                 f"вердикт False Positive получили {number(scan.fp.sum())}."
                 if len(scan) else "В этот период нет карточек с единственным правилом GL_FW_S04_Host_Port_Scan.")

    minutes = part.minutes.dropna()
    intervals = ["До 15 мин", "От 15 мин до 1 ч", "От 1 до 4 ч", "От 4 до 24 ч", "От 24 ч"]
    bucket = pd.cut(minutes, [0, 15, 60, 240, 1440, float("inf")], labels=intervals, right=False)
    distribution = bucket.value_counts(sort=False)
    time_fig = handling_bars(distribution, intervals, "Карточки", 270)
    if len(minutes):
        median, p90 = minutes.quantile([.5, .9])
        median_text = f"{median:.1f}".replace(".", ",")
        p90_text = f"{p90:.1f}".replace(".", ",")
        timing = f"Медиана {median_text} мин. 90% интервалов укладываются в {p90_text} мин."
    else:
        timing = "Нет карточек с измеримой длительностью."
    missing_dates = int(part.dates_missing.sum())
    coverage = (f"В расчёте {number(len(minutes))} из {number(len(part))} карточек. "
                f"Хотя бы одна дата отсутствует у {number(missing_dates)}, "
                f"порядок дат нарушен у {number(len(part) - len(minutes) - missing_dates)}.")
    return {"rule_chart": rules_fig, "noise_summary": noise, "scan_text": scan_text, "time_chart": time_fig,
            "time_summary": timing, "time_coverage": coverage}


def ranking(top, column):
    rates = column == "rate"
    ranked = top.sort_values(column, ascending=True)
    display_names = [name.replace("-", " ") for name in ranked.index]
    names = ["<br>".join(wrap(name, 30)) for name in display_names]
    labels = [
        percent(row.missing, row.cards) if rates else number(row.cards)
        for row in ranked.itertuples()
    ]
    hover = [
        f"{kind}<br>Карточек: {number(row.cards)}<br>Без вердикта: "
        f"{number(row.missing)} ({percent(row.missing, row.cards)})"
        for kind, row in zip(display_names, ranked.itertuples())
    ]
    fig = go.Figure(go.Bar(
        x=ranked[column].tolist(), y=names, orientation="h",
        marker_color=[BLUE if value == ranked[column].max() else "#9eafc1" for value in ranked[column]],
        text=labels, textposition="outside",
        cliponaxis=False, hovertext=hover, hovertemplate="%{hovertext}<extra></extra>",
    ))
    chart_style(fig, max(200, 56 * len(top) + 50))
    fig.update_xaxes(rangemode="tozero", showgrid=True, gridcolor="#e4ebee",
                     range=[0, 115 if rates else max(1, ranked[column].max() * 1.24)],
                     ticksuffix="%" if rates else "", tickformat=",~f", nticks=5)
    fig.update_yaxes(type="category", showgrid=False)
    return fig


def overview(counts, month):
    dates = counts["day"]
    data = counts if month == ALL else counts.loc[dates.dt.to_period("M") == pd.Period(month, freq="M")]
    if data.empty:
        raise ValueError(f"За выбранный период нет карточек: {month}")
    daily = data.groupby("day")["cards"].sum().sort_index()
    days = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    daily = daily.reindex(days, fill_value=0)
    by_kind = data.groupby("kind")[["cards", "missing"]].sum()
    top = by_kind.sort_values(["cards", "kind"], ascending=[False, True]).head(6).copy()
    top["rate"] = 100 * top["missing"] / top["cards"]
    leading = top.index[0]
    high = top["rate"].idxmax()
    leading_name = leading.replace("-", " ")
    high_name = high.replace("-", " ")
    total = int(daily.sum())
    missing = int(data["missing"].sum())
    peak_day = daily.idxmax()
    peak = int(daily.max())

    fig = go.Figure(go.Bar(
        x=[day.strftime("%Y-%m-%d") for day in days], y=daily.tolist(),
        marker=dict(color=[BLUE if value == peak else "#718da9" for value in daily], line_width=0),
        hovertemplate="%{x|%d.%m.%Y}<br>Карточек: %{y:,d}<extra></extra>",
    ))
    chart_style(fig, 235)
    fig.update_layout(bargap=0.08)
    fig.update_xaxes(type="date", tickformat="%d.%m.%y", nticks=min(7, len(days)), showgrid=False)
    fig.update_yaxes(rangemode="tozero", gridcolor="#e4ebee", title_text="Карточки", tickformat=",~f")
    return {
        "span": f"С {date_label(days[0])} по {date_label(days[-1])}",
        "volume": number(total),
        "missing_rate": percent(missing, total),
        "missing_detail": f"{number(missing)} из {number(total)}",
        "peak_value": number(peak),
        "peak_date": date_label(peak_day),
        "missing_metric": f"{number(missing)} · {percent(missing, total)}",
        "peak_text": f"Больше всего карточек поступило {date_label(peak_day)}. Всего {number(peak)}.",
        "kinds_text": f"На тип «{leading_name}» приходится {percent(top.at[leading, 'cards'], total)} карточек периода.",
        "rates_text": f"У типа «{high_name}» вердикт отсутствует у {number(top.at[high, 'missing'])} "
                      f"из {number(top.at[high, 'cards'])} карточек ({percent(top.at[high, 'missing'], top.at[high, 'cards'])}).",
        "flow_figure": fig, "types_figure": ranking(top, "cards"), "rates_figure": ranking(top, "rate"),
    }
