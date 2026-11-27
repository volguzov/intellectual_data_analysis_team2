"""Частота ложных тревог и парный пересчёт LightGBM v001 при удалении поля src."""

import argparse
from functools import lru_cache
from ipaddress import ip_address
from pathlib import Path

from explanations import COURSE, CourseModel, parse_json_list, read_csv

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from threadpoolctl import threadpool_limits


@lru_cache(maxsize=32768)
def address_group(value):
    addresses = [ip_address(item) for item in parse_json_list(value)]
    if not addresses:
        return "Пусто"
    external = [ip.is_global and not ip.is_multicast and not ip.is_reserved for ip in addresses]
    if all(external):
        return "Только внешние"
    return "Смешанные" if any(external) else "Без внешних"


def score_extract(path, model, batch_size=2048):
    chunks = []
    with threadpool_limits(limits=1):
        for batch in pq.ParquetFile(path).iter_batches(batch_size=batch_size):
            cards = batch.to_pandas()
            created = pd.to_datetime(cards["Создан"], format="ISO8601", utc=True, errors="raise")
            original = model.predict_proba(cards)
            removed = original.copy()
            present = cards.src.notna()
            if present.any():
                removed[present.to_numpy()] = model.predict_proba(cards.loc[present].assign(src=None))
            chunks.append(pd.DataFrame({
                "ID": pd.to_numeric(cards.EntityId, errors="raise"),
                "Создан": created, "Вердикт": cards["Вердикт"], "Источник": cards.src.map(address_group),
                "До": original, "Без src": removed,
                "В области модели": cards["Правило корреляции SIEM"].notna()
                                    & ~cards["Вендор из сием"].isin(model.cfg["dataset"]["filters"]["vendor_not_in"]),
            }))
    results = pd.concat(chunks, ignore_index=True).set_index("ID", verify_integrity=True)
    if results.index.hasnans or results["Создан"].isna().any() or not np.isfinite(results[["До", "Без src"]]).all().all():
        raise ValueError("Для каждой карточки нужны ID, время и обе оценки.")
    return results


def summarize(cards, threshold):
    incident = cards["Вердикт"].eq("Решён")
    negative = cards["Вердикт"].eq("False Positive")
    labeled = incident | negative
    before = cards["До"].ge(threshold)
    after = cards["Без src"].ge(threshold)
    added = after & ~before
    removed = before & ~after
    return {
        "Карточек": len(cards), "Размечено": int(labeled.sum()),
        "Инцидентов": int(incident.sum()), "Легитимных": int(negative.sum()),
        "TP до": int((incident & before).sum()), "FP до": int((negative & before).sum()),
        "TP после": int((incident & after).sum()), "FP после": int((negative & after).sum()),
        "Новые TP": int((incident & added).sum()), "Снятые TP": int((incident & removed).sum()),
        "Новые FP": int((negative & added).sum()), "Снятые FP": int((negative & removed).sum()),
        "Смен решений с меткой": int((labeled & (added | removed)).sum()),
        "Смен решений всего": int((added | removed).sum()),
    }


def post_training_changes(path, scores, model):
    end = pd.Timestamp(model.cfg["dataset"]["window"]["to"], tz="UTC") + pd.Timedelta(days=1)
    before = scores["До"].ge(model.threshold)
    after = scores["Без src"].ge(model.threshold)
    selected = scores.loc[scores["Создан"].ge(end)
                          & scores["Источник"].isin(["Только внешние", "Смешанные"])
                          & scores["Вердикт"].isin(["Решён", "False Positive"]) & before.ne(after)].copy()
    columns = ["EntityId", "Пиктограмма IRP", "Правило корреляции SIEM", "src", "dst"]
    chunks = []
    for batch in pq.ParquetFile(path).iter_batches(columns=columns, batch_size=8192):
        raw = batch.to_pandas()
        raw.index = pd.to_numeric(raw.pop("EntityId"), errors="raise")
        chunks.append(raw.loc[raw.index.isin(selected.index)])
    details = pd.concat(chunks, verify_integrity=True).loc[selected.index]
    return pd.DataFrame({
        "Дата UTC": selected["Создан"].dt.strftime("%Y-%m-%d"),
        "Вердикт": selected["Вердикт"],
        "После удаления src": np.where(after.loc[selected.index], "Тревога появилась", "Тревога исчезла"),
        "До, %": 100 * selected["До"], "Без src, %": 100 * selected["Без src"],
        "Категория": details["Пиктограмма IRP"].replace({"Атака на инфраструтуру": "Атака на инфраструктуру"}),
        "Правило SIEM": details["Правило корреляции SIEM"].map(lambda value: ", ".join(parse_json_list(value))),
        "Разных адресов src": details["src"].map(lambda value: len(set(parse_json_list(value)))),
        "Разных адресов dst": details["dst"].map(lambda value: len(set(parse_json_list(value)))),
    }).sort_index()


def compare_populations(scores, model):
    external = scores["Источник"].isin(["Только внешние", "Смешанные"])
    window = model.cfg["dataset"]["window"]
    start = pd.Timestamp(window["from"], tz="UTC")
    end = pd.Timestamp(window["to"], tz="UTC") + pd.Timedelta(days=1)
    created = scores["Создан"]
    groups = {
        "Вся выгрузка": scores,
        "После окна обучения, все источники": scores.loc[created.ge(end)],
        "Внешний src, весь период": scores.loc[external],
        "Внешний src, до окна обучения": scores.loc[external & created.lt(start)],
        "Внешний src, окно обучения": scores.loc[external & created.ge(start) & created.lt(end)],
        "Внешний src, после окна обучения": scores.loc[external & created.ge(end)],
        "Внешний src, после окна и с фильтрами модели": scores.loc[external & created.ge(end) & scores["В области модели"]],
        "Только внешние адреса": scores.loc[scores["Источник"].eq("Только внешние")],
        "Внешние и другие адреса вместе": scores.loc[scores["Источник"].eq("Смешанные")],
    }
    for name, file in [("320 учебных", "cards_practice.csv"), ("80 контрольных", "cards_control.csv")]:
        reference = read_csv(COURSE / "course_pack" / file).set_index("EntityId", verify_integrity=True)
        subset = scores.loc[reference.index]
        np.testing.assert_array_equal(subset["До"].ge(model.threshold), reference.outcome.isin(["TP", "FP"]))
        known = subset["Вердикт"].isin(["Решён", "False Positive"])
        if not known.all() or not subset["Вердикт"].eq("Решён").eq(reference.y_true).all():
            raise ValueError(f"Вердикты выгрузки отличаются от эталона {file}.")
        groups[f"Внешний src, {name}"] = subset.loc[external.loc[subset.index]]
    return pd.DataFrame.from_dict({name: summarize(cards, model.threshold) for name, cards in groups.items()}, orient="index")


def external_result_table():
    path = Path(__file__).with_name("source_removal_summary.csv")
    summary = pd.read_csv(path, encoding="utf-8-sig", index_col="Группа")
    rows = []
    for name, title in [("Внешний src, весь период", "Весь период"),
                        ("Внешний src, после окна обучения", "После обучения")]:
        group = summary.loc[name]
        legitimate = group["Легитимных"]
        fp = group["FP до"]
        percent = f"{100 * fp / legitimate:.1f}%".replace(".", ",") if legitimate else "нет данных"
        rows.append([title, f"{group['Размечено']:,}".replace(",", " "),
                     f"{fp:,} из {legitimate:,}".replace(",", " ") + f" ({percent})",
                     f"{group['Смен решений с меткой']:,}".replace(",", " ")])
    return pd.DataFrame(rows, columns=["Период", "Размечено", "Ложные тревоги среди легитимных", "Смен решений"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parquet", type=Path, help="Путь к полной выгрузке iavd_ml_dataset.parquet")
    parser.add_argument("--output", type=Path, help="Сохранить сводные количества по группам в CSV")
    parser.add_argument("--changes", type=Path, help="Сохранить карточки с изменившимся решением после окна обучения")
    args = parser.parse_args()
    model = CourseModel.from_registry("lgbm_gate1", 1)
    scores = score_extract(args.parquet, model)
    summary = compare_populations(scores, model)
    print(summary.to_string())
    if args.output:
        summary.to_csv(args.output, index_label="Группа", encoding="utf-8-sig")
    if args.changes:
        post_training_changes(args.parquet, scores, model).to_csv(
            args.changes, index_label="ID", encoding="utf-8-sig", float_format="%.6f")


if __name__ == "__main__":
    main()
