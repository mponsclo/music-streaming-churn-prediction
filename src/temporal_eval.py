"""
Temporal evaluation: train on Round 1, test on Round 2.

This mirrors the actual WSDM Cup 2018 competition setup:
- Round 1: users whose subscriptions expire in Feb 2017
- Round 2 (refresh): users whose subscriptions expire in March 2017

Features are computed independently for each round from their
respective data files so there is no information leakage across
the temporal boundary.
"""

import sys
from pathlib import Path

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    log_loss,
    roc_auc_score,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.paths import MODELS_DIR  # noqa: E402

DATA_DIR = PROJECT_ROOT / "data"

# Round 1 (original) paths
TRAIN_V1_PATH = DATA_DIR / "train.csv"
TXN_V1_PATH = DATA_DIR / "transactions.csv"
LOGS_V1_PATH = DATA_DIR / "user_logs 2.csv"
MEMBERS_PATH = DATA_DIR / "members_v3.csv"

# Round 2 (refresh) paths -- reuse the existing feature table
FEATURE_TABLE_V2 = PROJECT_ROOT / "outputs" / "feature_table.parquet"

CAT_FEATURES = ["gender", "city", "registered_via", "last_payment_method"]


def build_round1_features():
    """Build the same feature set from Round 1 data (original files)."""
    con = duckdb.connect()
    print("Building Round 1 features from original data files...")
    print("  (transactions.csv: 21.5M rows, user_logs: 16M Feb rows)")

    # The SQL mirrors our dbt models but points at original files
    # and filters user_logs to February 2017 (the last month before Round 1 prediction)
    features_sql = f"""
    WITH train AS (
        SELECT msno, is_churn
        FROM read_csv_auto('{TRAIN_V1_PATH}')
    ),

    members AS (
        SELECT
            msno,
            city,
            CASE WHEN bd BETWEEN 10 AND 70 THEN bd ELSE NULL END AS age,
            CASE
                WHEN gender = 'male' THEN 'male'
                WHEN gender = 'female' THEN 'female'
                ELSE 'unknown'
            END AS gender,
            registered_via,
            registration_init_time,
            datediff('day',
                strptime(registration_init_time::varchar, '%Y%m%d'),
                date '2017-02-01'
            ) AS tenure_days
        FROM read_csv_auto('{MEMBERS_PATH}')
    ),

    txns AS (
        SELECT
            msno,
            payment_method_id,
            payment_plan_days,
            plan_list_price,
            actual_amount_paid,
            is_auto_renew,
            is_cancel,
            try_strptime(transaction_date::varchar, '%Y%m%d') AS transaction_date,
            try_strptime(membership_expire_date::varchar, '%Y%m%d') AS membership_expire_date,
            CASE WHEN actual_amount_paid < plan_list_price THEN 1 ELSE 0 END AS has_discount
        FROM read_csv_auto('{TXN_V1_PATH}')
    ),

    last_txn AS (
        SELECT *, row_number() OVER (PARTITION BY msno ORDER BY transaction_date DESC) AS rn
        FROM txns
    ),

    txn_agg AS (
        SELECT
            msno,
            count(*) AS txn_count,
            sum(is_cancel) AS total_cancels,
            sum(is_auto_renew) AS total_auto_renews,
            sum(has_discount) AS total_discounts,
            max(membership_expire_date) AS latest_expire_date,
            max(transaction_date) AS latest_txn_date
        FROM txns
        GROUP BY msno
    ),

    txn_features AS (
        SELECT
            lt.msno,
            lt.payment_method_id AS last_payment_method,
            lt.is_auto_renew AS last_auto_renew,
            lt.is_cancel AS last_cancel,
            lt.payment_plan_days AS last_plan_days,
            lt.actual_amount_paid AS last_amount_paid,
            lt.plan_list_price AS last_list_price,
            lt.has_discount AS last_has_discount,
            CASE WHEN lt.payment_plan_days > 0
                THEN lt.plan_list_price::double / lt.payment_plan_days::double
                ELSE 0
            END AS last_price_per_day,
            agg.txn_count,
            agg.total_cancels,
            agg.total_auto_renews,
            agg.total_discounts,
            datediff('day', agg.latest_txn_date, agg.latest_expire_date) AS days_until_expiry
        FROM last_txn lt
        INNER JOIN txn_agg agg ON lt.msno = agg.msno
        WHERE lt.rn = 1
    ),

    -- Filter user_logs to February 2017 only (matching the Round 1 prediction window)
    logs AS (
        SELECT *
        FROM read_csv_auto('{LOGS_V1_PATH}')
        WHERE date BETWEEN 20170201 AND 20170228
    ),

    log_features AS (
        SELECT
            msno,
            count(DISTINCT date) AS active_days,
            coalesce(sum(total_secs), 0) AS total_secs,
            coalesce(sum(num_100), 0) AS complete_plays,
            coalesce(sum(num_unq), 0) AS unique_songs,
            coalesce(sum(num_25), 0) AS partial_plays,
            coalesce(sum(num_50), 0) AS half_plays,
            coalesce(sum(num_75), 0) AS three_quarter_plays,
            coalesce(sum(num_985), 0) AS near_complete_plays,
            coalesce(sum(num_25 + num_50 + num_75 + num_985 + num_100), 0) AS total_plays,
            CASE WHEN sum(num_25 + num_50 + num_75 + num_985 + num_100) > 0
                THEN sum(num_100)::double / sum(num_25 + num_50 + num_75 + num_985 + num_100)::double
                ELSE 0
            END AS completion_rate,
            CASE WHEN sum(num_25) > 0
                THEN sum(num_unq)::double / sum(num_25)::double
                ELSE 0
            END AS engagement_depth,
            CASE WHEN count(DISTINCT date) > 0
                THEN sum(total_secs)::double / count(DISTINCT date)::double
                ELSE 0
            END AS avg_daily_secs,
            CASE WHEN sum(num_unq) > 0
                THEN (sum(total_secs) / 60.0) / sum(num_unq)::double
                ELSE 0
            END AS mins_per_song
        FROM logs
        GROUP BY msno
    )

    SELECT
        t.msno,
        t.is_churn,

        CASE WHEN m.msno IS NOT NULL THEN 1 ELSE 0 END AS has_member_data,
        CASE WHEN tf.msno IS NOT NULL THEN 1 ELSE 0 END AS has_transaction_data,
        CASE WHEN m.age IS NOT NULL THEN 1 ELSE 0 END AS has_age,

        m.age, m.gender, m.city, m.registered_via, m.tenure_days,

        tf.last_payment_method, tf.last_auto_renew, tf.last_cancel,
        tf.last_plan_days, tf.last_amount_paid, tf.last_list_price,
        tf.last_has_discount, tf.last_price_per_day,

        coalesce(tf.txn_count, 0) AS txn_count,
        coalesce(tf.total_cancels, 0) AS total_cancels,
        coalesce(tf.total_auto_renews, 0) AS total_auto_renews,
        coalesce(tf.total_discounts, 0) AS total_discounts,
        tf.days_until_expiry,

        coalesce(lf.active_days, 0) AS active_days,
        coalesce(lf.total_secs, 0) AS total_secs,
        coalesce(lf.complete_plays, 0) AS complete_plays,
        coalesce(lf.unique_songs, 0) AS unique_songs,
        coalesce(lf.partial_plays, 0) AS partial_plays,
        coalesce(lf.half_plays, 0) AS half_plays,
        coalesce(lf.three_quarter_plays, 0) AS three_quarter_plays,
        coalesce(lf.near_complete_plays, 0) AS near_complete_plays,
        coalesce(lf.total_plays, 0) AS total_plays,
        coalesce(lf.completion_rate, 0) AS completion_rate,
        coalesce(lf.engagement_depth, 0) AS engagement_depth,
        coalesce(lf.avg_daily_secs, 0) AS avg_daily_secs,
        coalesce(lf.mins_per_song, 0) AS mins_per_song

    FROM train t
    LEFT JOIN members m ON t.msno = m.msno
    LEFT JOIN txn_features tf ON t.msno = tf.msno
    LEFT JOIN log_features lf ON t.msno = lf.msno
    """

    df = con.execute(features_sql).fetchdf()
    print(f"  Round 1 features: {df.shape[0]:,} rows x {df.shape[1]} cols")
    print(f"  Churn rate: {df['is_churn'].mean() * 100:.2f}%")
    return df


def compute_metrics(y_true, y_prob, name=""):
    """Compute standard classification metrics."""
    ll = log_loss(y_true, y_prob)
    roc = roc_auc_score(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)

    thresholds = np.arange(0.05, 0.95, 0.01)
    f1s = [f1_score(y_true, (y_prob >= t).astype(int)) for t in thresholds]
    best_f1 = max(f1s)
    best_thresh = thresholds[np.argmax(f1s)]

    if name:
        print(f"\n  {name}")
        print(f"  {'=' * 50}")
    print(f"  log_loss:     {ll:.5f}")
    print(f"  roc_auc:      {roc:.5f}")
    print(f"  pr_auc:       {pr_auc:.5f}")
    print(f"  f1:           {best_f1:.5f} @{best_thresh:.2f}")
    return {"log_loss": ll, "roc_auc": roc, "pr_auc": pr_auc, "f1": best_f1}


def main():
    # 1. Build Round 1 features
    train_df = build_round1_features()

    # 2. Load Round 2 features (already built by dbt)
    test_df = pd.read_parquet(FEATURE_TABLE_V2)
    print(f"\nRound 2 features: {test_df.shape[0]:,} rows x {test_df.shape[1]} cols")
    print(f"  Churn rate: {test_df['is_churn'].mean() * 100:.2f}%")

    # 3. Prepare train/test
    feature_cols = [c for c in train_df.columns if c not in ("msno", "is_churn")]
    X_train = train_df[feature_cols].copy()
    y_train = train_df["is_churn"]
    X_test = test_df[feature_cols].copy()
    y_test = test_df["is_churn"]

    for col in CAT_FEATURES:
        X_train[col] = X_train[col].astype("category")
        X_test[col] = X_test[col].astype("category")

    # 4. Train LightGBM with the best params from Optuna
    import json

    study_path = MODELS_DIR / "optuna_study.json"
    with open(study_path) as f:
        study = json.load(f)

    scale_pos = (y_train == 0).sum() / (y_train == 1).sum()
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "verbosity": -1,
        "feature_pre_filter": False,
        "scale_pos_weight": scale_pos,
        "boosting_type": "gbdt",
        **study["best_params"],
    }
    best_round = study["best_round"]

    print(f"\nTraining LightGBM on Round 1 ({len(X_train):,} users)...")
    dtrain = lgb.Dataset(X_train, label=y_train, categorical_feature=CAT_FEATURES)
    model = lgb.train(params, dtrain, num_boost_round=best_round, callbacks=[lgb.log_evaluation(200)])

    # 5. Evaluate on Round 2 (temporal holdout)
    y_prob = model.predict(X_test)

    print(f"\n{'=' * 60}")
    print("  TEMPORAL EVALUATION: Train Round 1 -> Test Round 2")
    print(f"  Train: {len(X_train):,} users (Feb 2017, {y_train.mean() * 100:.1f}% churn)")
    print(f"  Test:  {len(X_test):,} users (Mar 2017, {y_test.mean() * 100:.1f}% churn)")
    print(f"{'=' * 60}")
    metrics = compute_metrics(y_test, y_prob, name="LightGBM (temporal holdout)")

    print("\n  Compare to random split: Log Loss=0.073, ROC-AUC=0.993")
    print(f"  Temporal holdout:        Log Loss={metrics['log_loss']:.3f}, ROC-AUC={metrics['roc_auc']:.3f}")


if __name__ == "__main__":
    main()
