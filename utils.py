import os
import random
import shutil

import geopandas as gpd
import rasterio
import pandas as pd

from rasterio.io import MemoryFile
from owslib.wms import WebMapService
from shapely.geometry import box
from ultralytics import YOLO
from concurrent.futures import ThreadPoolExecutor

from joblib import Parallel, delayed
from tqdm import tqdm

# Funzione per scaricare e processare una singola tile
def process_tile(i, bbox, wms, layer_name, model, output_dir):
    width, height = 1024, 1024  # Riduci a 512x512 per accelerare

    # Richiesta dell'immagine WMS
    response = wms.getmap(
        layers=[layer_name],
        srs='EPSG:32633',
        bbox=bbox,
        size=(width, height),
        format='image/geotiff',
        transparent=True
    )

    # Salva l'immagine temporanea e processa con YOLO
    temp_tif = os.path.join(output_dir, f"wms_image_{i}.tif")
    with MemoryFile(response.read()) as memfile:
        with memfile.open() as dataset:
            # Salva il raster su disco
            with rasterio.open(temp_tif, "w", **dataset.profile) as dest:
                dest.write(dataset.read())

    # Effettua la predizione
    result = model(temp_tif, conf = 0.5)
    predictions = [[temp_tif, result]]
    
    # Crea lo shapefile
    output_shp = os.path.join(output_dir, f"bbox_{i}.shp")
    create_shapefile(predictions, output_shp)

    # Rimuovi il file temporaneo
    os.remove(temp_tif)

# Funzione per convertire pixel in coordinate geografiche
def pixel_to_geo(pixel_x, pixel_y, transform):
    x, y = rasterio.transform.xy(transform, pixel_y, pixel_x, offset='center')
    return x, y

# Funzione per creare shapefile dalle predizioni
def create_shapefile(predictions, output_shp):
    geometries = []
    scores = []
    crs = None

    for tile_path, result in predictions:
        with rasterio.open(tile_path) as src:
            if crs is None:
                crs = src.crs
            for pred in result[0].boxes.xyxy.cpu().numpy():
                xmin, ymin = pixel_to_geo(pred[0], pred[1], src.transform)
                xmax, ymax = pixel_to_geo(pred[2], pred[3], src.transform)
                geometries.append(box(xmin, ymin, xmax, ymax))
            for pred in result[0].boxes.conf.cpu().numpy():
                scores.append(pred)
            

    if len(geometries) > 0:
        gdf = gpd.GeoDataFrame({'geometry': geometries, 'confidence': scores}, crs=crs) #gpd.GeoDataFrame(geometry=geometries, crs=crs)
        gdf.to_file(output_shp)




def create_train_val_split(images_dir, output_dir, train_ratio=0.8):
    """
    Crea i file train.txt e val.txt per YOLO a partire da una directory di immagini.

    Args:
        images_dir (str): Directory contenente tutte le immagini.
        output_dir (str): Directory dove salvare train.txt e val.txt.
        train_ratio (float): Percentuale di immagini da usare per il training (default: 0.8).
    """
    # Ottieni tutti i file immagine
    image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png', '.tif'))]

    # Percorsi completi per train.txt e val.txt
    train_file = os.path.join(output_dir, "train.txt")
    val_file = os.path.join(output_dir, "val.txt")

    # Mescola e suddividi le immagini
    random.shuffle(image_files)
    split_index = int(len(image_files) * train_ratio)
    train_files = image_files[:split_index]
    val_files = image_files[split_index:]

    # Scrivi i percorsi completi nei file train.txt e val.txt
    with open(train_file, "w") as train_f:
        for img in train_files:
            train_f.write(os.path.join(images_dir, img) + "\n")

    with open(val_file, "w") as val_f:
        for img in val_files:
            val_f.write(os.path.join(images_dir, img) + "\n")

    print(f"Creati: {len(train_files)} immagini in train.txt, {len(val_files)} immagini in val.txt")

# Processamento parallelo delle tile
def process_all_tiles_in_parallel(gdf, wms, layer_name, model, output_dir):
    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = []
        for i in range(len(gdf.geometry)):
            bbox = gdf.geometry[i].bounds
            futures.append(executor.submit(process_tile, i, bbox, wms, layer_name, model, output_dir))
        
        for future in futures:
            future.result()

# Funzione per leggere e filtrare uno shapefile
def read_and_filter_shapefile(filepath):
    gdf = gpd.read_file(filepath)
    filtered_gdf = gdf
    #filtered_gdf = gdf[gdf.intersects(nuts_boundary)]  # Filtra per intersezione con il poligono di delimitazione
    return filtered_gdf if not filtered_gdf.empty else None
