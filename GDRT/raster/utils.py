import logging
import shutil

import numpy as np
import pyproj
import rasterio as rio
from rasterio import warp
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


def load_geospatial_crop(
    input_file, region_of_interest, target_CRS=None, target_GSD=None
):
    with rio.open(input_file) as dataset:
        input_CRS = dataset.crs

    if target_CRS is None:
        target_CRS = input_CRS

    if input_CRS != target_CRS:
        # We need to reproject the input data into the desired CRS
        # For efficiency sake, we should limit this only to the needed region
        # but this means we need to first transform the ROI into the target_CRS, then compute the bounding rectangle,
        # then transform this bounding rectangle back into the input CRS, then find the bounding rectangle around
        # this region. This addresses the fact that the axes of the two CRS are not necessarily aligned
        ROI_envelope_in_target_CRS = region_of_interest.to_crs(target_CRS).envelope
        ROI_envelope_back_in_input_CRS = ROI_envelope_in_target_CRS.to_crs(
            input_CRS
        ).envelope

        crop_in_original_CRS, crop_transform = load_geospatial_crop(
            input_file=input_file,
            region_of_interest=ROI_envelope_back_in_input_CRS,
            target_CRS=input_CRS,
            target_GSD=target_GSD,
        )
        dataset_transform = None
        raise NotImplementedError()
    else:
        geospatial_bounds = region_of_interest.to_crs(target_CRS).bounds
        minx = np.squeeze(geospatial_bounds.minx.values)
        miny = np.squeeze(geospatial_bounds.miny.values)
        maxx = np.squeeze(geospatial_bounds.maxx.values)
        maxy = np.squeeze(geospatial_bounds.maxy.values)

        with rio.open(input_file) as dataset:
            logging.info(dataset.transform)

            scale_factor = 1 if target_GSD is None else dataset.transform.a / target_GSD

            logging.info(f"minx: {minx},  miny: {miny}, maxx: {maxx}, maxy: {maxy}")
            ((min_px, max_px), (min_py, max_py)) = dataset.index(
                [minx, maxx], [miny, maxy]
            )

            # TODO figure out why x width is swapped
            window = rio.windows.Window.from_slices((min_py, max_py), (max_px, min_px))
            out_shape = (
                dataset.count,
                int(window.height * scale_factor),
                int(window.width * scale_factor),
            )

            window_raster = dataset.read(
                window=window,
                out_shape=out_shape,
                resampling=warp.Resampling.bilinear,
            )
            window_transform = dataset.window_transform(window)

            rescaling = rio.transform.Affine.scale(
                window.width / window_raster.shape[2],
                window.height / window_raster.shape[1],
            )
            window_transform = window_transform * rescaling
            dataset_transform = dataset.transform

    window_image = reshape_as_image(window_raster)
    return window_image, np.array(window_transform).reshape(3,3), np.array(dataset_transform).reshape(3,3)
