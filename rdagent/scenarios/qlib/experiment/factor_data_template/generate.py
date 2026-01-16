import qlib

import os

provider_uri = (
    os.environ.get("QLIB_PROVIDER_URI", "").strip()
    or os.environ.get("AISTOCK_QLIB_PROVIDER_URI", "").strip()
    or "/mnt/f/Dev/AIstock/qlib_bin/qlib_bin_20251209"
)
provider_uri = provider_uri.strip() or "~/.qlib/qlib_data/cn_data"

qlib.init(provider_uri=provider_uri)

from qlib.data import D

instruments = D.instruments()
fields = ["$open", "$close", "$high", "$low", "$volume", "$factor"]
data = D.features(instruments, fields, freq="day").swaplevel().sort_index().loc["2008-12-29":].sort_index()

data = data.rename(columns={c: c.lstrip("$") for c in data.columns})

data.to_hdf("./daily_pv_all.h5", key="data")


fields = ["$open", "$close", "$high", "$low", "$volume", "$factor"]
data = (
    (
        D.features(instruments, fields, start_time="2018-01-01", end_time="2019-12-31", freq="day")
        .swaplevel()
        .sort_index()
    )
    .swaplevel()
    .loc[data.reset_index()["instrument"].unique()[:100]]
    .swaplevel()
    .sort_index()
)

data = data.rename(columns={c: c.lstrip("$") for c in data.columns})
data.to_hdf("./daily_pv_debug.h5", key="data")
