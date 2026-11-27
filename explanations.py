"""Практика 3.7. Инференс и сверка объяснений учебного LightGBM."""

import json
import sys
from itertools import product
from pathlib import Path
from textwrap import wrap
from zipfile import ZipFile

COURSE = Path(__file__).resolve().parent / "data" / "IAVD_3.7_ML_team"
sys.path.insert(0, str(COURSE))
try:
    # Учебный пакет загружает LightGBM до PyArrow на Windows.
    from course_ml.core import read_csv
    from course_ml.data import FREQ_FIELDS, ips, is_private, parse_json_list, prepare
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
SOURCE_ADDRESSES = {"Частный": "10.0.0.1", "Внешний": "1.1.1.1"}
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


def source_threshold_chart(cards, threshold):
    fig = go.Figure()
    for verdict, marker, color in [("Инцидент", "circle", BLUE), ("Ложная тревога", "diamond", "#ac491c")]:
        group = cards.loc[cards["Новая тревога"].eq(verdict)]
        fig.add_scatter(
            x=group["До, %"], y=group["Карточка"], mode="markers", name=verdict,
            marker=dict(size=11, symbol=marker, color=color),
            customdata=group[["До порога, п. п.", "После, %"]].to_numpy(),
            hovertemplate="№ %{y}<br>До подстановки: %{x:.4f}%<br>До порога: %{customdata[0]:.4f} п. п."
                          "<br>После подстановки: %{customdata[1]:.4f}%<extra></extra>",
        )
    chart_style(fig, 455)
    fig.add_vline(x=threshold, line_dash="dot", line_color="#364b5e")
    fig.add_annotation(x=threshold, y=1, yref="paper", xanchor="right", yanchor="bottom",
                       text=f"Порог {fmt(threshold, 4)}%", showarrow=False)
    fig.update_yaxes(type="category", categoryorder="array", categoryarray=cards["Карточка"].tolist(),
                     autorange="reversed", title_text="Карточка")
    fig.update_xaxes(range=[0, threshold * 1.055], title_text="Оценка до подстановки, %")
    fig.update_layout(showlegend=True, legend=dict(orientation="h", x=0, y=-0.21),
                      margin=dict(l=10, r=22, t=30, b=85))
    return fig


def case_notes(raw, reference, private_share_median):
    cohort = reference[["outcome", "src_is_private"]].join(
        raw[["src", "Пиктограмма IRP"]], how="left", validate="one_to_one")
    addresses = cohort.src.map(ips)
    missing = addresses.str.len().eq(0)
    category = cohort["Пиктограмма IRP"]
    same = category.eq(category.at[523952])
    missed = cohort.outcome.eq("FN")
    found = cohort.outcome.eq("TP")
    external = addresses.map(lambda items: bool(items) and not any(is_private(ip) for ip in items))
    peers = cohort.loc[category.eq(category.at[527002]) & external]
    groups = peers.groupby("outcome").src_is_private.agg(["size", "mean"])
    fp, tp = groups.loc["FP"], groups.loc["TP"]
    return {
        523952: ("Адрес источника отсутствует. Вместо неизвестной доли частных IP модель использовала "
                 f"обучающую медиану {fmt(private_share_median, 1)}. Вклад признака в балл составил "
                 f"{fmt(reference.at[523952, 'src_is_private'])}. "
                 f"В категории «{category.at[523952]}» адрес отсутствует у "
                 f"{(same & missed & missing).sum()} из {(same & missed).sum()} пропущенных инцидентов. "
                 f"Обнаруженных инцидентов этой категории в учебной выборке: {(same & found).sum()}. "
                 f"По всем категориям адрес отсутствует у {(missed & missing).sum()} из {missed.sum()} "
                 f"пропущенных и у {(found & missing).sum()} из {found.sum()} обнаруженных. "
                 "Для этой карточки сверим адрес с исходным событием."),
        527002: ("В карточке указан один внешний адрес. Доля частных IP равна нулю. "
                 f"Этот признак повысил балл на {fmt(reference.at[527002, 'src_is_private'])}. "
                 f"В категории «{category.at[527002]}» среди карточек только с внешними адресами источника "
                 f"средний вклад доли частных IP составил {fmt(fp['mean'])} для ложных тревог "
                 f"({int(fp['size'])}) и {fmt(tp['mean'])} для верных ({int(tp['size'])}). "
                 "Признак в среднем повышал балл в обеих группах. Сравним модели с ним и без него на другом периоде при одном пороге."),
    }


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
        imputer = self.model.preproc.named_transformers_["numeric"].named_steps["imputer"]
        share_index = list(imputer.feature_names_in_).index("src_is_private")
        self.private_share_median = float(imputer.statistics_[share_index])
        self.source_scores = self._missing_source_scores()
        self.notes = case_notes(self.raw, self.reference, self.private_share_median)
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
        first, last = 100 * points.iloc[0], 100 * points.iloc[-1]
        self.pdp_summary = (f"При заданной длине {points.index[0]:g} и {points.index[-1]:g} слов средние оценки составляют "
                            f"{fmt(first, 2)}% и {fmt(last, 2)}% соответственно. "
                            f"Разница {fmt(last - first, 2)} процентного пункта.")
        self.pdp_fig = go.Figure(go.Scatter(x=points.index.tolist(),
                                            y=(100 * points).tolist(), mode="lines+markers"))
        chart_style(self.pdp_fig, 260)
        self.pdp_fig.update_xaxes(title_text="Слов в тексте алерта")
        self.pdp_fig.update_yaxes(title_text="Средняя оценка модели, %")
        self.options = {f"{key} · {OUTCOMES[r.outcome]}": key for key, r in self.team.iterrows()}

    def _missing_source_scores(self):
        cohort = self.raw.loc[self.reference.index]
        missing = cohort.loc[cohort.src.map(ips).str.len().eq(0)]
        decisions, original_scores = self.model.predict(missing)
        expected = self.reference.loc[missing.index, "outcome"].isin(["TP", "FP"])
        if not np.array_equal(decisions, expected):
            raise ValueError("Исходные решения по карточкам без адреса расходятся с эталоном.")
        scores = pd.DataFrame({"Исходная карточка": original_scores}, index=missing.index)
        for kind, address in SOURCE_ADDRESSES.items():
            scenario = missing.assign(src=json.dumps([address]))
            scores[kind] = self.model.predict_proba(scenario)
        return scores

    def source_address_scenarios(self):
        scores = self.source_scores.loc[523952]
        addresses = ["Исходная карточка, адрес отсутствует"]
        addresses.extend(f"{kind} · {address}" for kind, address in SOURCE_ADDRESSES.items())
        return pd.DataFrame({
            "Адрес источника": addresses,
            "Оценка модели": [fmt(100 * score, 6) + "%" for score in scores],
            "Решение": ["Тревога" if score >= self.model.threshold else "Без тревоги" for score in scores],
        })

    def source_address_transitions(self):
        decisions = self.source_scores.ge(self.model.threshold)
        original = decisions["Исходная карточка"]
        incident = self.reference.loc[original.index, "y_true"].eq(1)
        rows = [
            f"Изменилось решений из {len(original)}",
            f"Обнаружено из {int((incident & ~original).sum())} прежних пропусков",
            "Появилось пропусков инцидентов",
            "Появилось ложных тревог",
            "Снято ложных тревог",
        ]
        table = pd.DataFrame({"Что изменилось": rows})
        for kind, address in SOURCE_ADDRESSES.items():
            appeared = decisions[kind] & ~original
            removed = original & ~decisions[kind]
            table[f"{kind} · {address}"] = [
                (appeared | removed).sum(), (appeared & incident).sum(), (removed & incident).sum(),
                (appeared & ~incident).sum(), (removed & ~incident).sum(),
            ]
        return table

    def external_address_crossings(self):
        threshold = self.model.threshold
        eligible = self.source_scores.loc[self.source_scores["Исходная карточка"].lt(threshold)]
        crossed = eligible["Внешний"].ge(threshold)
        scores = eligible.loc[crossed]
        raw = self.raw.loc[scores.index]
        truth = self.reference.loc[scores.index, "y_true"]
        cards = pd.DataFrame({
            "Карточка": scores.index.astype(str),
            "Новая тревога": truth.map({1: "Инцидент", 0: "Ложная тревога"}),
            "До, %": 100 * scores["Исходная карточка"],
            "До порога, п. п.": 100 * (threshold - scores["Исходная карточка"]),
            "После, %": 100 * scores["Внешний"],
            "Правило SIEM": raw["Правило корреляции SIEM"].map(lambda value: ", ".join(parse_json_list(value))),
        }).sort_values(["Новая тревога", "До порога, п. п."]).reset_index(drop=True)
        chart = source_threshold_chart(cards, 100 * threshold)
        other = eligible.loc[~crossed, "Исходная карточка"]
        summary = (f"До подстановки медиана оценки у {len(cards)} новых тревог равна "
                   f"{fmt(cards['До, %'].median(), 4)}%. У остальных {len(other)} карточек с пустым src, "
                   f"которые остались без тревоги, медиана равна {fmt(100 * other.median(), 4)}%.")
        for column in ["До, %", "До порога, п. п.", "После, %"]:
            cards[column] = cards[column].map(lambda value: fmt(value, 4))
        return chart, cards, summary

    def malware_rule_shap(self, crossing_ids):
        rule = "TTS_10.1_Malware_Found_And_Not_Deleted"
        candidates = self.raw.loc[crossing_ids]
        matches = candidates["Правило корреляции SIEM"].map(lambda value: rule in parse_json_list(value))
        rows = []
        for key in candidates.index[matches]:
            original = self.raw.loc[[key]]
            changed = original.assign(src=json.dumps([SOURCE_ADDRESSES["Внешний"]]))
            _, p0, margin0, base0, before = self._card_contributions(original)
            _, p1, margin1, base1, after = self._card_contributions(changed)
            delta = after - before
            expected = self.source_scores.loc[key, ["Исходная карточка", "Внешний"]]
            if (abs(base1 - base0) > 1e-9 or abs(delta.sum() - (margin1 - margin0)) > 5e-7
                    or not np.allclose([p0, p1], expected, rtol=0, atol=1e-10)):
                raise ValueError(f"Вклады для карточки {key} расходятся с пересчётом прогноза.")
            top = delta[delta.gt(0)].nlargest(3).index
            parts = pd.DataFrame({"До": before, "После": after, "Изменение": delta})
            shown = parts.loc[top].rename(index=FIELD_LABELS)
            shown.loc[f"Остальные поля ({len(parts) - len(top)})"] = parts.drop(top).sum()
            for field, values in shown.iterrows():
                row = {"Карточка": str(key), "Признак": field}
                row.update({name: fmt(value, 4) for name, value in values.items()})
                rows.append(row)
        return pd.DataFrame(rows, columns=["Карточка", "Признак", "До", "После", "Изменение"])

    def private_share_scenarios(self, card_ids):
        cards = self.raw.loc[card_ids]
        _, inputs = self._model_inputs(cards)
        if not inputs["src_is_private"].isna().all() or self.private_share_median != 1.0:
            raise ValueError("Для этого опыта нужна неизвестная доля частных IP с медианой заполнения 1.")
        changed = inputs.assign(src_is_private=0.0)
        scores = self._input_probabilities(changed)
        reference = self.source_scores.loc[cards.index]
        return pd.DataFrame({
            "Карточка": cards.index.astype(str),
            "Исходная оценка": [fmt(100 * p, 4) + "%" for p in reference["Исходная карточка"]],
            "Только доля 0": [fmt(100 * p, 4) + "%" for p in scores],
            "Решение при доле 0": ["Тревога" if p >= self.model.threshold else "Без тревоги" for p in scores],
            "С адресом 1.1.1.1": [fmt(100 * p, 4) + "%" for p in reference["Внешний"]],
        })

    def address_combinations_526902(self):
        card = self.raw.loc[[526902]]
        _, inputs = self._model_inputs(card)
        pair = pd.concat([inputs, inputs], ignore_index=True)
        pair.at[1, "src"] = SOURCE_ADDRESSES["Внешний"]
        rows = []
        for length, count in product((0, 1), repeat=2):
            scenario = pair.assign(src_is_private=0.0, len_src=length, ip_count_src=count)
            scores = self._input_probabilities(scenario)
            cells = []
            for score in scores:
                decision = "тревога" if score >= self.model.threshold else "без тревоги"
                cells.append(f"{fmt(100 * score, 4)}% · {decision}")
            rows.append([length, count, *cells])
        return pd.DataFrame(rows, columns=[
            "Элементов в src", "Распознанных IP", "Пустой src", "src = 1.1.1.1",
        ])

    def external_addresses_526902(self):
        _, inputs = self._model_inputs(self.raw.loc[[526902]])
        addresses = [inputs.at[526902, "src"], SOURCE_ADDRESSES["Внешний"], "1.1.1.2",
                     "1.0.0.1", "8.8.8.8", "9.9.9.9", "208.67.222.222"]
        scenarios = pd.concat([inputs] * len(addresses), ignore_index=True).assign(
            src=addresses, src_is_private=0.0, len_src=0, ip_count_src=0)
        scores = self._input_probabilities(scenarios)
        return pd.DataFrame({
            "Значение src": ["Пустое поле", *addresses[1:]],
            "Оценка": [fmt(100 * score, 4) + "%" for score in scores],
            "Решение": ["Тревога" if score >= self.model.threshold else "Без тревоги" for score in scores],
        })

    def _input_probabilities(self, inputs):
        matrix = np.asarray(self.model.preproc.transform(inputs), dtype=np.float32)
        return self.model.model.predict_proba(matrix, num_threads=1)[:, 1]

    def _model_inputs(self, cards):
        prepared, _, _ = prepare(cards, self.model.cfg)
        for name, source in FREQ_FIELDS.items():
            prepared[name] = prepared[source].astype(str).map(self.model.freq_maps[source]).fillna(0.0)
        return prepared, feature_frame(prepared, self.model.cfg)

    def _card_contributions(self, card):
        prepared, inputs = self._model_inputs(card)
        matrix = np.asarray(self.model.preproc.transform(inputs), dtype=np.float32)
        estimator = self.model.model
        p = float(estimator.predict_proba(matrix, num_threads=1)[0, 1])
        raw_score = float(estimator.predict(matrix, raw_score=True, num_threads=1)[0])
        contrib = estimator.predict(matrix, pred_contrib=True, num_threads=1)[0]
        base = float(contrib[-1])
        fields = pd.Series(contrib[:-1]).groupby(self.mapping.source_column.to_numpy()).sum()
        return prepared, p, raw_score, base, fields

    def explain(self, label):
        key = self.options[label]
        card = self.raw.loc[[key]]
        prepared, p, raw_score, base, fields = self._card_contributions(card)
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
                shown = (f"Доля неизвестна. До масштабирования подставлено {fmt(self.private_share_median, 1)}"
                         if field == "src_is_private" else "нет значения")
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
        share_value = (f"доля неизвестна, перед масштабированием подставлена медиана {fmt(self.private_share_median, 1)}"
                       if pd.isna(private_share) else fmt(100 * private_share, 1) + "%")
        leading = top.index[0]
        direction = "повысило" if top.iloc[0] > 0 else "снизило"
        return {
            "x_show_source_test": key == 523952,
            "x_outcome": OUTCOMES[outcome].capitalize(),
            "x_verdict": "Аналитик подтвердил инцидент" if actual.y_true else "Аналитик признал событие легитимным",
            "x_probability": fmt(p * 100, 5).rstrip("0").rstrip(",") + "%",
            "x_decision": "Модель подала сигнал тревоги" if p >= self.model.threshold else "Модель оставила карточку без тревоги",
            "x_threshold": fmt(self.model.threshold * 100, 4) + "%",
            "x_note": self.notes.get(key, f"Поле «{FIELD_LABELS.get(leading, leading)}» дало наибольший вклад "
                                f"по величине и {direction} балл модели на {fmt(abs(top.iloc[0]))}."),
            "x_address_check": f"Поле src: {source_value}. Доля частных адресов (src_is_private): {share_value}.",
            "x_water": water, "x_top": bars(top, "Вклад SHAP",
                                           ["#b45529" if v > 0 else TEAL for v in top]),
            "x_table": pd.DataFrame(observations),
            "x_check": f"База {fmt(base, 6)} + сумма 73 вкладов {fmt(fields.sum(), 6)} "
                       f"= {fmt(raw_score, 6)}. Расхождение с исходным баллом модели {margin_error:.2g}. "
                       f"После логистического преобразования расхождение с оценкой {prob_error:.2g}.",
        }
