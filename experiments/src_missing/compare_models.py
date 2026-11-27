"""Парное сравнение четырёх наборов признаков на размеченных карточках SOC."""
import argparse
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--course", type=Path, default=DATA / "IAVD_3.7_ML_team",
                    help="Папка распакованного комплекта практики 3.7")
parser.add_argument("--data", type=Path, default=DATA / "iavd_ml_dataset.parquet",
                    help="Полная выгрузка для обучения")
parser.add_argument("--output", type=Path, default=DATA / "src_missing_metrics.csv",
                    help="Путь для сводки повторного расчёта")
args = parser.parse_args()
if not (args.course / "models" / "lgbm_gate1__v001" / "run.json").is_file():
    parser.error(f"Не найден комплект модели в {args.course}. Распакуйте архив по инструкции в README.")
if not args.data.is_file():
    parser.error(f"Не найден {args.data}. Скачайте полную выгрузку по ссылке в README.")
sys.path.insert(0, str(args.course.resolve()))

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import average_precision_score
from threadpoolctl import threadpool_limits
from course_ml import data, train
from course_ml.core import read_csv
from course_ml.inference import CourseModel


def read_window(path, cfg):
    dc = cfg["dataset"]
    frames = []
    start = pd.Timestamp(dc["window"]["from"], tz="UTC")
    last = pd.Timestamp(dc["window"]["to"], tz="UTC") + pd.Timedelta(days=1)
    for batch in pq.ParquetFile(path).iter_batches(batch_size=4096):
        raw = batch.to_pandas()
        day = pd.to_datetime(raw[dc["time"]["column"]], utc=True, format="ISO8601")
        keep = day.ge(start) & raw[dc["target"]["column"]].isin(dc["filters"]["verdict_in"])
        keep &= ~raw["Вендор из сием"].isin(dc["filters"]["vendor_not_in"])
        if dc["filters"]["rule_not_null"]:
            keep &= raw["Правило корреляции SIEM"].notna()
        if keep.any():
            frames.append(raw.loc[keep].copy())
    raw = pd.concat(frames, ignore_index=True)
    if not raw.EntityId.is_unique or raw.EntityId.isna().any():
        raise ValueError("Нужна одна строка с известным номером на карточку.")
    dates = pd.to_datetime(raw[dc["time"]["column"]], utc=True, format="ISO8601")
    return raw.loc[dates.lt(last)].reset_index(drop=True), raw.loc[dates.ge(last)].reset_index(drop=True)


def matrix(advisor, prepared):
    frame = train.feature_frame(prepared, advisor.cfg)
    return np.concatenate([np.asarray(advisor.preproc.transform(frame.iloc[i:i + 2048]), dtype=np.float32)
                           for i in range(0, len(frame), 2048)])


def layout(base, missing, private_idx, keep_private, add_flag):
    inputs = base if keep_private else np.delete(base, private_idx, axis=1)
    if add_flag:
        inputs = np.column_stack((inputs, missing.astype(np.float32)))
    return inputs


def record(rows, seed, variant, group, policy, threshold, y, score, baseline):
    alarm = score >= threshold
    positive = y == 1
    tp = np.sum(positive & alarm)
    fp = np.sum(~positive & alarm)
    rows.append(dict(seed=seed, variant=variant, group=group, policy=policy, threshold=threshold,
                     n=len(y), incidents=positive.sum(), tp=tp, fn=positive.sum()-tp, fp=fp,
                     tn=(~positive).sum()-fp, alerts=tp+fp, average_precision=average_precision_score(y, score),
                     gained_tp=np.sum(positive & alarm & ~baseline), lost_tp=np.sum(positive & ~alarm & baseline),
                     added_fp=np.sum(~positive & alarm & ~baseline), removed_fp=np.sum(~positive & ~alarm & baseline)))


def main():
    advisor = CourseModel(args.course / "models" / "lgbm_gate1__v001")
    ref = advisor.cfg
    window, future = read_window(args.data, ref)
    prepared, y, created = data.prepare(window, ref)
    parts = train.make_split(prepared, y, created, ref)
    maps = train.add_freq_features(prepared, parts, ref)
    for field, values in maps.items():
        expected = advisor.freq_maps[field]
        if values.keys() != expected.keys() or any(abs(value-expected[k]) > 1e-12 for k, value in values.items()):
            raise ValueError("Отбор для обучения расходится с картами частот эталона.")
    post, y_post, dates = data.prepare(future, ref)
    for col, field in data.FREQ_FIELDS.items():
        post[col] = post[field].astype(str).map(maps[field]).fillna(0.0)
    mapping = read_csv(advisor.dir / "feature_mapping.csv")
    indices = mapping.loc[mapping.source_column.eq("src_is_private"), "feature_index"]
    if len(indices) != 1:
        raise ValueError("Ожидается один вход модели для доли частных IP.")
    private = int(indices.iloc[0])
    missing = prepared.src.fillna("").str.strip().eq("").to_numpy()
    missing_post = post.src.fillna("").str.strip().eq("").to_numpy()
    print("Непустой/пустой адрес: несовпадения с len_src == 0 за два окна",
          np.sum(missing != prepared.len_src.eq(0)), np.sum(missing_post != post.len_src.eq(0)), flush=True)
    print("Окно курса", {key: len(ix) for key, ix in parts.items()}, "после обучения",len(post),
          dates.min(),dates.max(),flush=True)
    print("Преобразование входов общим препроцессором v001",flush=True)
    with threadpool_limits(limits=1):
        x_window = matrix(advisor, prepared)
        x_future = matrix(advisor, post)
    assert x_window.shape[1] == len(mapping) == advisor.model.n_features_in_
    train_idx, val_idx, test_idx = (parts[key] for key in ("train","val","test"))
    test = train.metrics_row(y.iloc[test_idx], advisor.model.booster_.predict(x_window[test_idx],num_threads=4),
                             advisor.threshold, "контроль")
    if [test[k] for k in ("tp","fp","fn","tn")] != [390,107,24,2652]:
        raise ValueError(f"Контрольное разбиение отличается от паспорта модели: {test}")
    official = advisor.model.booster_.predict(x_future,num_threads=4)
    all_y = np.concatenate([y.iloc[val_idx], y.iloc[test_idx], y_post])
    n_val, n_test = len(val_idx),len(test_idx)
    start = n_val + n_test
    months = dates.dt.strftime("%Y-%m")
    groups = {"validation": np.arange(n_val), "in_window_test": n_val+np.arange(n_test),
              "post": start+np.arange(len(post)), "post_missing": start+np.flatnonzero(missing_post),
              "post_present": start+np.flatnonzero(~missing_post)}
    for month in sorted(months.unique()):
        groups[month] = start+np.flatnonzero(months.eq(month))
    compare_x = np.concatenate([x_window[val_idx], x_window[test_idx], x_future])
    compare_missing = np.concatenate([missing[val_idx],missing[test_idx],missing_post])
    training = x_window[train_idx]
    del x_window,x_future
    variants = [("baseline",True,False),("flag",True,True),("no_private",False,False),("replace",False,True)]
    rows = []
    for seed in (42,43,44):
        bases = {}
        for label,keep,add in variants:
            params = dict(ref["model"]["params"], random_state=seed)
            model = lgb.LGBMClassifier(**params)
            model.fit(layout(training, missing[train_idx], private, keep, add), y.iloc[train_idx].to_numpy())
            scores = model.booster_.predict(layout(compare_x,compare_missing,private,keep,add), num_threads=4)
            if not np.isfinite(scores).all():
                raise ValueError("Модель вернула неопределённые оценки.")
            if seed==42 and label=="baseline":
                print("Отличие от опубликованной модели после переобучения: max |Δp|",
                      np.max(np.abs(scores[start:]-official)), "разных решений",
                      np.sum((scores[start:]>=advisor.threshold)!=(official>=advisor.threshold)),flush=True)
            policies = {"fixed": advisor.threshold, "validation_F2": train.best_f2_threshold(all_y[:n_val],scores[:n_val])}
            for policy,threshold in policies.items():
                if label=="baseline":
                    bases[policy] = scores>=threshold
                for group,ix in groups.items():
                    record(rows,seed,label,group,policy,threshold,all_y[ix],scores[ix],bases[policy][ix])
            print(seed,label,"OK", "validation threshold",policies["validation_F2"],flush=True)
    output = pd.DataFrame(rows)
    if args.output.exists() and args.output.is_dir():
        raise ValueError("Укажите путь к файлу CSV.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=args.output.parent, suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        output.to_csv(temporary, index=False, encoding="utf-8-sig")
        temporary.replace(args.output)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Сохранено {len(output)} строк: {args.output}", flush=True)


if __name__ == "__main__":
    main()
