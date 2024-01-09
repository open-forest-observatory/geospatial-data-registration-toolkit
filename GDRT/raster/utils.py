import logging
import shutil

import pyproj
import rasterio as rio
from rasterio import warp
import numpy as np
from rasterio.plot import reshape_as_image



# https://stackoverflow.com/questions/60288953/how-to-change-the-crs-of-a-raster-with-rasterio
def reproject_raster(in_path, out_path, out_crs=pyproj.CRS.from_epsg(4326)):
    """ """
    logging.warning("Starting to reproject raster")
    # reproject raster to project crs
    with rio.open(in_path) as src:
        src_crs = src.crs
        if src_crs == out_crs:
            logging.warning("Copying instead since source and target CRS are identical")
            shutil.copy(in_path, out_path)
            return

        transform, width, height = rio.warp.calculate_default_transform(
            src_crs, out_crs, src.width, src.height, *src.bounds
        )
        kwargs = src.meta.copy()

        kwargs.update(
            {"crs": out_crs, "transform": transform, "width": width, "height": height}
        )

        with rio.open(out_path, "w", **kwargs) as dst:
            for i in range(1, src.count + 1):
                print(f"Reprojected band {i}")
                warp.reproject(
                    source=rio.band(src, i),
                    destination=rio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=out_crs,
                    resampling=rio.warp.Resampling.nearest,
                )
    logging.warn("Done reprojecting raster")


def load_geospatial_crop(input_file, region_of_interest, target_CRS=None, target_GSD=None):

    geospatial_bounds = region_of_interest.bounds
    minx = np.squeeze(geospatial_bounds.minx.values)
    miny = np.squeeze(geospatial_bounds.miny.values)
    maxx = np.squeeze(geospatial_bounds.maxx.values)
    maxy = np.squeeze(geospatial_bounds.maxy.values)
    
    with rio.open(input_file) as dataset:
        logging.info(dataset.transform)

        scale_factor = 1 if target_GSD is None else dataset.transform.a / target_GSD 

        logging.info(f"minx: {minx},  miny: {miny}, maxx: {maxx}, maxy: {maxy}")
        ((min_px, max_px), (min_py, max_py)) = dataset.index([minx, maxx], [miny, maxy])

        # TODO figure out why x width is swapped
        window = rio.windows.Window.from_slices((min_py, max_py), (max_px, min_px))
        windowed_band = dataset.read(window=window,
            out_shape=(dataset.count,
                int(dataset.height * scale_factor),
                int(dataset.width * scale_factor)
            ),
            resampling=warp.Resampling.bilinear
        )
        windowed_band = reshape_as_image(windowed_band)
    return windowed_band
