"""Generate a tiny synthetic schema fixture and run the real ingestion pipeline.

This is a data-engineering demonstration, not simulated atmospheric truth.
Deliberate issues: absent ERA5 hour 01:00, one negative PM value, one missing
sensor longitude. These remain visible in the final QC rather than repaired.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import yaml

from prepare_modeling_dataset import run


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,default=Path("configs/modeling_demo.yaml"))
    config=yaml.safe_load(parser.parse_args().config.read_text())
    paths=[Path(config["era5_files"][0]),Path(config["sensor_csv"])]
    if len(config["era5_files"]) != 1 or any(p.exists() for p in paths) or Path(config["output_dir"]).exists():
        raise FileExistsError("Demo needs one new ERA5 file, new sensor file and new processed directory.")
    for p in paths:
        p.parent.mkdir(parents=True,exist_ok=True)
    hours=np.array([0,2,3])
    lat,lon=np.array([43.5,43.25,43.]),np.array([76.5,77.,77.5])
    hh,yy,xx=np.meshgrid(hours,lat,lon,indexing="ij")
    dims=("valid_time","latitude","longitude")
    ds=xr.Dataset(dict(u10=(dims,2*(xx-76)+3*(yy-43)+hh,{"units":"m s-1"}),
        v10=(dims,np.full(hh.shape,-1.),{"units":"m s-1"}),
        t2m=(dims,273.+hh+(xx-76),{"units":"K"}),
        blh=(dims,300.+100*hh,{"units":"m"})),
        coords=dict(valid_time=np.datetime64("2020-01-01T00:00:00")+hours*np.timedelta64(1,"h"),latitude=lat,longitude=lon),
        attrs=dict(source="synthetic test fixture; not operational ERA5"))
    ds.to_netcdf(paths[0])
    rng=np.random.default_rng(config["experiment"]["seed"])
    rows=[]
    for hour in (0,1,2):
        for i,(longitude,latitude) in enumerate(((76.75,43.2),(77.,43.25),(77.2,43.35))):
            rows.append(dict(sensor_id=f"demo_{i}",timestamp=f"2020-01-01T0{hour}:30:00Z",
                longitude=longitude,latitude=latitude,pm25=20.+rng.normal()))
    rows[0]["pm25"]=-1.
    rows[-1]["longitude"]=np.nan
    pd.DataFrame(rows).to_csv(paths[1],index=False)
    print(json.dumps(run(config),indent=2))


if __name__ == "__main__":
    main()
