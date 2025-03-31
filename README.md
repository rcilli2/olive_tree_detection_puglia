# Olive Tree Detection in Puglia, Italy using YOLO11n and RGB VHR aerial imagery.

This project implements an end-to-end pipeline for large-scale olive tree detection in Puglia (Southern Italy) featuring:
 ## YOLO11n Customized Model
 - YOLO11n model training on 0.2m-resolution aerial imagery from AGEA (Agenzia Erogazioni in Agricoltura, ndr.)
## Parallelized Large-Area Inference
 - Distributed processing of 1TB+ imagery via ThreadPoolExecutor
 - Grid-based analysis using 1024×1024 px tiles (≈200m×200m at 0.2m/px)
 - On-the-fly WMS image retrieval → detection → automatic cleanup workflow
 - Output: Georeferenced shapefiles per tile with confidence scores
 - Fast Inference: 24 hour processing time for the whole Apulia region on a NVIDIA A100-PCIE-40GB (total area 20,000 Km^2)
## Spatial Data Consolidation
 - High-performance merging of tile predictions into unified regional shapefiles consisting of 50M olive trees.
 - Topology-preserving aggregation for seamless polygon boundaries (in the works).
 - Memory-efficient processing chain for potential national-scale deployment

The methodology has been successfully tested in the BAT (Barletta-Andria-Trani) and northern Bari areas, but it currently shows limitations in Foggia and Salento due to insufficient training data representation.
![Esempio di rilevamento ulivi](assets/sample_detection.jpg) <!-- Aggiungi una foto esemplificativa se disponibile -->

Interactive Results
Detection results for northern Bari are available on Google Earth Engine:
🔗 Interactive GEE Map Link:
https://code.earthengine.google.com/2c92db76cdf42ebbc958e84e63c8b58c?hideCode=true

Technical Details
- Spatial resolution: 0.2m/px (commercial satellite data)
- Tile size: 1024×1024 px (≈200m×200m)
- Model: Custom-trained YOLO11n
- Performance (BAT area):
    - Precision: 93%
    - Recall: 89%
    - mAP@0.5: 93%
    - R^2 : 0.90
    - Median Absolute Error: 11.4 Trees per Hectare

## Known Limitations
Performance degradation occurs in: (1) Foggia/Salento regions (limited training samples), (2) densely vegetated areas (false positives), and (3) Xylella-infected zones where canopy damage complicates detection. 

## Scientific Publication
📄 A paper describing the methodology is under review:

"Automated Olive Grove Classification and Tree Counting from Very High Resolution Aerial Imagery using Deep Learning", Pantaleo, E. et al. (under review)

# olive_tree_detection_puglia
# olive_tree_detection_puglia
