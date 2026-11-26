"""Практика 3.7. Инференс и сверка объяснений учебного LightGBM."""

import json
import sys
from pathlib import Path
from textwrap import wrap
from zipfile import ZipFile

COURSE = Path(__file__).resolve().parent / "data" / "IAVD_3.7_ML_team"
sys.path.insert(0, str(COURSE))
try:
    # Учебный пакет загружает LightGBM до PyArrow на Windows.
    from course_ml.core import read_csv
    from course_ml.data import FREQ_FIELDS, ips, parse_json_list, prepare
    from course_ml.inference import CourseModel
    from course_ml.train import feature_frame
finally:
    sys.path.remove(str(COURSE))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.special import expit, logit

from overview import BLUE, TEAL, chart_style


OUTCOMES = {"FN": "пропуск", "FP": "ложная тревога", "TP": "верная тревога", "TN": "верное отклонение"}
NOTES = {
    523952: "Адрес источника пропущен. Тем не менее признак доли частных адресов снизил оценку. "
            "Стоит проверить, как часто адрес пропущен у других инцидентов этой категории.",
    527002: "Адрес источника находится вне частных диапазонов по правилу курса. "
            "Вклад этого признака повысил оценку. Стоит сравнить его с вкладом у верных тревог той же категории.",
}

FIELD_LABELS = {
    "Hostname атакующего": "Имя хоста источника", "Hostname Цели": "Имя хоста цели",
    "src": "Адрес источника", "dst": "Адрес назначения", "src_is_private": "Доля частных IP источника",
    "freq_Наименование": "Частота наименования", "Создан": "Время создания", "alert_words": "Число слов в алерте",
    "event_age_days": "Возраст события, дни", "create_lag_s": "Задержка создания, с",
    "cmd_entropy": "Энтропия командной строки", "CMDLINE": "Командная строка", "host_is_server": "Признак сервера",
    "object.name": "Имя объекта", "categoryHigh": "Категория верхнего уровня", "categoryMain": "Основная категория",
    "len_Значение Инцидента (группировка)": "Значений в группе инцидента",
    "len_Значение Атаки (группировка)": "Значений в группе атаки",
}


def fmt(value, precision=3):
    return f"{value:.{precision}f}".replace(".", ",").replace("-", "−")


def axis_labels(fields, width):
    return ["<br>".join(wrap(FIELD_LABELS.get(field, field), width)) for field in fields]


def bars(values, axis, color):
    bars = go.Figure(go.Bar(x=values.to_list(), y=values.index.to_list(), orientation="h",
                            marker_color=color, hovertemplate="%{y}<br>%{x:.4f}<extra></extra>"))
    chart_style(bars, 48 + 34 * len(values))
    bars.update_yaxes(autorange="reversed", tickmode="array", tickvals=values.index.tolist(),
                      ticktext=axis_labels(values.index, 22), tickfont_size=12)
    bars.update_xaxes(title_text=axis, zeroline=True, zerolinecolor="#697b86")
    return bars


class Explanations:
    def __init__(self):
        pack = COURSE / "course_pack"
        self.model = CourseModel.from_registry("lgbm_gate1", 1)
        self.mapping = read_csv(self.model.dir / "feature_mapping.csv").sort_values("feature_index")
        cards = read_csv(pack / "cards_practice.csv")
        self.team = cards.loc[cards.team.eq(2)].sort_values("rank_in_team").set_index("EntityId", verify_integrity=True)
        self.raw = pd.read_parquet(pack / "cards_raw.parquet")
        self.raw.index = self.raw.EntityId.astype(int)
        self.reference = read_csv(pack / "shap_cards.csv").set_index("EntityId", verify_integrity=True)
        fields = self.reference.drop(columns=["y_true", "proba", "base", "outcome"])
        if (self.team.outcome.value_counts().to_dict() != dict.fromkeys(OUTCOMES, 2)
                or len(fields.columns) != 73 or set(fields) != set(self.mapping.source_column)):
            raise ValueError("Состав набора отличается от учебной версии 3.7.")
        top = fields.abs().mean().nlargest(5)
        self.global_fig = bars(top, "Средний |SHAP|", BLUE)
        archive = COURSE.parent / "3.3_XAI_данные.zip"
        with ZipFile(archive) as z:
            spec = json.loads(z.read("spec.json").decode("utf-8-sig"))
            with z.open("pdp_numeric.csv") as f:
                pdp = pd.read_csv(f, sep=";")
        if spec["run_id"] != self.model.run_info["run_id"]:
            raise ValueError("PDP относится к другому прогону модели.")
        points = pdp.loc[pdp.feature.eq("alert_words")].groupby("raw_value")["mean_proba"].mean()
        self.pdp_fig = go.Figure(go.Scatter(x=points.index.tolist(),
                                            y=(100 * points).tolist(), mode="lines+markers"))
        chart_style(self.pdp_fig, 260)
        self.pdp_fig.update_xaxes(title_text="Слов в тексте алерта")
        self.pdp_fig.update_yaxes(title_text="Средняя оценка модели, %")
        self.options = {f"{key} · {OUTCOMES[r.outcome]}": key for key, r in self.team.iterrows()}

    def explain(self, label):
        key = self.options[label]
        card = self.raw.loc[[key]]
        prepared, _, _ = prepare(card, self.model.cfg)
        for name, source in FREQ_FIELDS.items():
            prepared[name] = prepared[source].astype(str).map(self.model.freq_maps[source]).fillna(0.0)
        inputs = feature_frame(prepared, self.model.cfg)
        matrix = np.asarray(self.model.preproc.transform(inputs), dtype=np.float32)
        estimator = self.model.model
        p = float(estimator.predict_proba(matrix, num_threads=1)[0, 1])
        raw_score = float(estimator.predict(matrix, raw_score=True, num_threads=1)[0])
        contrib = estimator.predict(matrix, pred_contrib=True, num_threads=1)[0]
        base = float(contrib[-1])
        fields = pd.Series(contrib[:-1]).groupby(self.mapping.source_column.to_numpy()).sum()
        actual = self.team.loc[key]
        outcome = ("TP" if actual.y_true else "FP") if p >= self.model.threshold else ("FN" if actual.y_true else "TN")
        archive = self.reference.loc[key]
        margin_error = abs(base + fields.sum() - raw_score)
        prob_error = abs(expit(raw_score) - p)
        if (outcome != actual.outcome or margin_error > 5e-7 or prob_error > 1e-6
                or abs(p - archive.proba) > 1e-6 or abs(base - archive.base) > 1e-6
                or (fields - archive[fields.index].astype(float)).abs().max() > 1e-6):
            raise ValueError(f"Контрольные данные для карточки {key} расходятся с моделью.")

        top = fields.reindex(fields.abs().nlargest(10).index)
        tail = fields.drop(top.index).sum()
        labels = ["База", *top.index, "Остальные 63", "Итог"]
        water = go.Figure(go.Waterfall(orientation="h", y=labels, x=[base, *top.values, tail, 0],
                                      customdata=[fmt(v) for v in (base, *top.values, tail, raw_score)],
                                      hovertemplate="%{y}: %{customdata}<extra></extra>",
                                      measure=["absolute", *(["relative"] * 11), "total"],
                                      increasing=dict(marker_color="#b45529"), decreasing=dict(marker_color=TEAL),
                                      totals=dict(marker_color=BLUE)))
        chart_style(water, 548)
        boundary = float(logit(self.model.threshold))
        water.add_vline(x=boundary, line=dict(color="#23394f", dash="dash", width=2))
        water.add_annotation(x=boundary, y=1, yref="paper", yshift=6, yanchor="bottom",
                             text=f"Порог {fmt(boundary, 2)}", showarrow=False)
        water.add_scatter(x=[raw_score], y=["Итог"], mode="markers",
                          marker=dict(size=10, color="#192e46", line=dict(color="white", width=1.5)),
                          hovertemplate=f"Итог модели {fmt(raw_score, 3)}<extra></extra>")
        water.update_layout(margin_t=34)
        water.update_yaxes(autorange="reversed", type="category", tickmode="array", tickvals=labels,
                           ticktext=axis_labels(labels, 24), tickfont_size=12)
        water.update_xaxes(title_text="Логарифм шансов")
        observations = []
        for field, value in top.items():
            derived = field not in card.columns
            v = prepared[field].iloc[0] if derived else card[field].iloc[0]
            if pd.isna(v):
                shown = "нет значения"
            elif derived and isinstance(v, (float, np.floating)):
                shown = f"{v:.6g}"
            elif field in ("Команда", "CMDLINE", "Алерт"):
                shown = f"Элементов в исходном поле: {len(parse_json_list(v))}"
            else:
                shown = str(v)[:130]
            observations.append({"Поле": field, "Происхождение": "Вычислено" if derived else "Карточка",
                                 "Значение": shown, "Вклад SHAP": fmt(value)})
        addresses = ips(card["src"].iloc[0])
        private_share = prepared["src_is_private"].iloc[0]
        source_value = ", ".join(addresses) if addresses else "адрес не указан"
        share_value = "нет значения" if pd.isna(private_share) else fmt(100 * private_share, 1) + "%"
        leading = top.index[0]
        direction = "повысило" if top.iloc[0] > 0 else "снизило"
        return {
            "x_outcome": OUTCOMES[outcome].capitalize(),
            "x_verdict": "Аналитик подтвердил инцидент" if actual.y_true else "Аналитик признал событие легитимным",
            "x_probability": fmt(p * 100, 5).rstrip("0").rstrip(",") + "%",
            "x_decision": "Модель подала сигнал тревоги" if p >= self.model.threshold else "Модель оставила карточку без тревоги",
            "x_threshold": fmt(self.model.threshold * 100, 4) + "%",
            "x_note": NOTES.get(key, f"Поле «{FIELD_LABELS.get(leading, leading)}» дало наибольший вклад "
                                f"по величине и {direction} балл модели на {fmt(abs(top.iloc[0]))}."),
            "x_address_check": f"Поле src: {source_value}. Доля частных адресов (src_is_private): {share_value}.",
            "x_water": water, "x_top": bars(top, "Вклад SHAP",
                                           ["#b45529" if v > 0 else TEAL for v in top]),
            "x_table": pd.DataFrame(observations),
            "x_check": f"База {fmt(base, 6)} + сумма 73 вкладов {fmt(fields.sum(), 6)} "
                       f"= {fmt(raw_score, 6)}. Расхождение с исходным баллом модели {margin_error:.2g}. "
                       f"После логистического преобразования расхождение с оценкой {prob_error:.2g}.",
        }
