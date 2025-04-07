from datetime import datetime
import math
import numpy as np

def normalize_timestamp(date: datetime):
    # From https://github.com/Clay-foundation/model/blob/main/finetune/embedder/how-to-embed.ipynb
    week = date.isocalendar().week * 2 * np.pi / 52
    hour = date.hour * 2 * np.pi / 24

    return (math.sin(week), math.cos(week)), (math.sin(hour), math.cos(hour))


# Prep lat/lon embedding using the
def normalize_latlon(lat: float, lon: float):
    # From https://github.com/Clay-foundation/model/blob/main/finetune/embedder/how-to-embed.ipynb
    lat = lat * np.pi / 180
    lon = lon * np.pi / 180

    return (math.sin(lat), math.cos(lat)), (math.sin(lon), math.cos(lon))