"""NSL-KDD loading, encoding, scaling and splitting.

Categorical columns (protocol_type, service, flag) are ordinal-encoded rather than
one-hot so that each of the 41 original features stays one bit in a hawk's mask.
That keeps selected-feature names meaningful (e.g. `src_bytes`) on the website.
"""

from __future__ import annotations

import argparse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder

RAW_DIR = Path(__file__).parent / "data" / "raw"
TRAIN_FILE = "KDDTrain+.txt"
TEST_FILE = "KDDTest+.txt"
MIRROR = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/"

FEATURES = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
]
COLUMNS = FEATURES + ["label", "difficulty"]
CATEGORICAL = ["protocol_type", "service", "flag"]

ATTACK_CATEGORY = {
    "normal": "normal",
    # DoS
    "back": "dos", "land": "dos", "neptune": "dos", "pod": "dos", "smurf": "dos",
    "teardrop": "dos", "apache2": "dos", "udpstorm": "dos", "processtable": "dos",
    "mailbomb": "dos",
    # Probe
    "satan": "probe", "ipsweep": "probe", "nmap": "probe", "portsweep": "probe",
    "mscan": "probe", "saint": "probe",
    # R2L
    "guess_passwd": "r2l", "ftp_write": "r2l", "imap": "r2l", "phf": "r2l",
    "multihop": "r2l", "warezmaster": "r2l", "warezclient": "r2l", "spy": "r2l",
    "xlock": "r2l", "xsnoop": "r2l", "snmpguess": "r2l", "snmpgetattack": "r2l",
    "httptunnel": "r2l", "sendmail": "r2l", "named": "r2l", "worm": "r2l",
    # U2R
    "buffer_overflow": "u2r", "loadmodule": "u2r", "rootkit": "u2r", "perl": "u2r",
    "sqlattack": "u2r", "xterm": "u2r", "ps": "u2r",
}
CLASSES_MULTI = ["normal", "dos", "probe", "r2l", "u2r"]


@dataclass
class Dataset:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]
    class_names: list[str]


def download(force: bool = False) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for name in (TRAIN_FILE, TEST_FILE):
        dest = RAW_DIR / name
        if dest.exists() and not force:
            print(f"have {dest}")
            continue
        print(f"downloading {name} ...")
        urllib.request.urlretrieve(MIRROR + name, dest)


def _read(name: str) -> pd.DataFrame:
    path = RAW_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; run `python data.py --download`")
    return pd.read_csv(path, header=None, names=COLUMNS)


def _labels(df: pd.DataFrame, task: str) -> np.ndarray:
    if task == "binary":
        return (df["label"] != "normal").astype(np.int64).to_numpy()
    cats = df["label"].map(ATTACK_CATEGORY)
    if cats.isna().any():
        unknown = sorted(df.loc[cats.isna(), "label"].unique())
        raise ValueError(f"unmapped attack labels: {unknown}")
    return cats.map(CLASSES_MULTI.index).to_numpy(np.int64)


def load(task: str = "binary", val_size: float = 0.2, seed: int = 42) -> Dataset:
    """Train/val come from KDDTrain+ (stratified), test is the official KDDTest+."""
    if task not in ("binary", "multi"):
        raise ValueError("task must be 'binary' or 'multi'")
    train_df, test_df = _read(TRAIN_FILE), _read(TEST_FILE)

    # Test set has services/flags unseen in train -> encode them as -1, then scaled.
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    enc.fit(train_df[CATEGORICAL])
    for df in (train_df, test_df):
        df[CATEGORICAL] = enc.transform(df[CATEGORICAL])

    X_full = train_df[FEATURES].to_numpy(np.float64)
    y_full = _labels(train_df, task)
    X_test = test_df[FEATURES].to_numpy(np.float64)
    y_test = _labels(test_df, task)

    X_train, X_val, y_train, y_val = train_test_split(
        X_full, y_full, test_size=val_size, stratify=y_full, random_state=seed
    )

    # src_bytes/dst_bytes are heavy-tailed; log1p before min-max keeps them usable for KNN.
    for X in (X_train, X_val, X_test):
        X[:] = np.log1p(np.clip(X, 0, None))
    scaler = MinMaxScaler().fit(X_train)
    X_train, X_val, X_test = (np.clip(scaler.transform(X), 0, 1) for X in (X_train, X_val, X_test))

    classes = ["normal", "attack"] if task == "binary" else CLASSES_MULTI
    return Dataset(X_train, y_train, X_val, y_val, X_test, y_test, list(FEATURES), classes)


def baseline(task: str = "binary") -> None:
    """Reference numbers: KNN(k=5) on all 41 features."""
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.neighbors import KNeighborsClassifier

    ds = load(task)
    clf = KNeighborsClassifier(n_neighbors=5, n_jobs=-1).fit(ds.X_train, ds.y_train)
    for split, X, y in (("val", ds.X_val, ds.y_val), ("test", ds.X_test, ds.y_test)):
        pred = clf.predict(X)
        print(f"{task:6s} {split:4s} acc={accuracy_score(y, pred):.4f} "
              f"macroF1={f1_score(y, pred, average='macro'):.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--baseline", action="store_true")
    ap.add_argument("--task", default="binary", choices=["binary", "multi"])
    args = ap.parse_args()
    if args.download:
        download()
    if args.baseline:
        baseline(args.task)
    if not (args.download or args.baseline):
        ds = load(args.task)
        print(f"train {ds.X_train.shape}  val {ds.X_val.shape}  test {ds.X_test.shape}")
        print("class counts (train):", np.bincount(ds.y_train))
