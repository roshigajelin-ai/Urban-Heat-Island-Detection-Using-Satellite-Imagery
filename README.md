# Explainable Vision Transformer for Urban Heat Island Detection Using Satellite Imagery

An explainable deep learning prototype that uses Landsat 8 satellite imagery to classify spatial patches into four **relative thermal intensity classes**. The project combines geospatial data preprocessing, spectral feature engineering, Vision Transformers, attention-based visualizations, and an interactive web application.

## Overview

Urban Heat Islands (UHIs) occur when urban areas experience higher temperatures than their surrounding areas. Built-up surfaces such as roads, concrete, and rooftops can absorb and retain more heat than vegetation and moist soil.

This project explores how satellite imagery and deep learning can help identify spatial patterns associated with relative thermal intensity.

The system processes Landsat 8 imagery, derives vegetation and built-up indices, generates relative thermal labels, and trains a Vision Transformer (ViT) to classify satellite image patches. A FastAPI backend connects the trained model to a React frontend for prediction and map-based visualization.

> **Important scientific note:** The four output classes represent relative thermal intensities derived from Landsat Band 10. They are not calibrated Land Surface Temperature (LST) measurements in degrees Celsius, nor independently validated Urban Heat Island classifications.

## Objectives

- Process and analyze Landsat 8 multispectral satellite imagery.
- Calculate NDVI and NDBI for vegetation and built-up surface context.
- Generate four relative thermal intensity classes from Band 10.
- Extract image patches and prepare eight-channel model inputs.
- Train and evaluate a Vision Transformer using PyTorch.
- Explore attention rollout for model interpretability.
- Develop a FastAPI backend for prediction and heatmap data.
- Visualize spatial results through a React and Leaflet interface.
- Identify limitations and opportunities for scientifically validated thermal analysis.

## Technology Stack

| Component | Technology |
|---|---|
| Programming language | Python |
| Satellite data | Landsat 8 GeoTIFF imagery |
| Geospatial processing | Rasterio, NumPy |
| Deep learning | PyTorch |
| Model architecture | Vision Transformer (ViT) |
| Explainable AI | Attention rollout |
| Backend | FastAPI |
| Frontend | React |
| Map visualization | Leaflet |
| Data visualization | Heatmaps and attention overlays |

## Dataset Description

The project uses Landsat 8 multispectral and thermal infrared raster data stored in GeoTIFF format.

The six spectral bands used as model inputs are:

- Band 2 — Blue
- Band 3 — Green
- Band 4 — Red
- Band 5 — Near Infrared (NIR)
- Band 6 — Short-Wave Infrared 1 (SWIR 1)
- Band 7 — Short-Wave Infrared 2 (SWIR 2)

Two additional features are derived from these bands:

- **NDVI:** Normalized Difference Vegetation Index
- **NDBI:** Normalized Difference Built-up Index

Band 10 is used to generate the target labels and is excluded from the model input to reduce direct label leakage.

### Feature Engineering

**Normalized Difference Vegetation Index (NDVI)**

NDVI provides an indication of vegetation characteristics.

NDVI = (B5 − B4) / (B5 + B4)

**Normalized Difference Built-up Index (NDBI)**

NDBI helps highlight built-up surface patterns.

NDBI = (B6 − B5) / (B6 + B5)

Both indices require appropriate reflectance preprocessing for scientifically reliable interpretation. Values calculated directly from uncalibrated digital numbers should be treated as exploratory.

## Methodology

The project workflow consists of the following stages:

1. **Raster inspection:** Examine band dimensions, coordinate reference systems, pixel values, and invalid data.
2. **Feature engineering:** Calculate NDVI and NDBI.
3. **Thermal label generation:** Mask invalid Band 10 pixels and derive four relative intensity classes.
4. **Feature stacking:** Combine Bands 2–7, NDVI, and NDBI into an eight-channel feature raster.
5. **Patch extraction:** Divide the raster into 64 × 64 pixel patches using a stride of 64 pixels.
6. **Label assignment:** Assign each eligible patch its dominant valid thermal class.
7. **Data splitting:** Create spatially separated training, validation, and test sets.
8. **Normalization:** Calculate normalization statistics from the training data.
9. **Model training:** Train a Vision Transformer to classify the four target classes.
10. **Evaluation:** Measure performance on the held-out test set.
11. **Explainability:** Generate attention rollout visualizations.
12. **Deployment and visualization:** Serve predictions through FastAPI and display spatial results through React and Leaflet.

## Vision Transformer Architecture

The model uses a compact Vision Transformer adapted for multispectral satellite imagery.

| Parameter | Configuration |
|---|---|
| Input patch | 64 × 64 pixels |
| Input channels | 8 |
| Transformer token size | 16 × 16 pixels |
| Spatial image tokens | 16 |
| Embedding dimension | 128 |
| Attention heads | 4 |
| Encoder layers | 4 |
| MLP dimension | 256 |
| Output classes | 4 |
| Training environment | CPU-based |

Each 64 × 64 input patch is divided into sixteen 16 × 16 spatial tokens. These tokens are embedded into feature vectors and processed through transformer encoder layers. The classification head produces scores for the four relative thermal intensity classes.

## Relative Thermal Intensity Classes

The target classes are generated from the distribution of valid Band 10 values in the selected scene.

| Class | Interpretation |
|---|---|
| Low | Lower relative thermal intensity |
| Moderate | Moderately low-to-middle relative intensity |
| High | Higher relative thermal intensity |
| Very High | Highest relative intensity group |

The class boundaries are scene-dependent. They should not be interpreted as universal temperature thresholds.

## Experimental Results

The current experimental configuration produced the following results:

| Metric | Result |
|---|---:|
| Total extracted patches | 9,750 |
| Training patches | 6,844 |
| Validation patches | 1,454 |
| Test patches | 1,452 |
| Correctly classified test patches | 856 |
| Test accuracy | 58.95% |

### Target Class Distribution

| Class | Number of patches |
|---|---:|
| Low | 2,307 |
| Moderate | 2,359 |
| High | 2,366 |
| Very High | 2,718 |
| **Total** | **9,750** |

The model achieved 58.95% test accuracy on the derived relative thermal labels. This indicates its agreement with those proxy labels on the held-out test patches.

Further evaluation using a confusion matrix, per-class precision and recall, macro-F1, balanced accuracy, and baseline models would provide a more complete assessment. Performance on independent scenes and cities also remains to be established.

## Explainable AI

The project investigates attention rollout to visualize how the Vision Transformer distributes attention across image tokens.

The process combines attention information across transformer layers, incorporates residual connections, and produces a coarse spatial map that can be resized and overlaid on the input patch.

These visualizations provide an exploratory view of the model's internal attention patterns. They do not, by themselves, establish causal explanations for a prediction. Future evaluation could compare attention rollout with occlusion sensitivity and gradient-based explanation methods.

## Web Application

The project includes a web interface designed to connect deep learning predictions with geospatial visualization.

### FastAPI Backend

The backend is designed to:

- Provide a health-check endpoint.
- Load the trained model and training normalization statistics.
- Return patch-level predictions and class scores.
- Supply geospatial heatmap points derived from the thermal class raster.

### React Frontend

The frontend is designed to:

- Display heatmap points on an interactive Leaflet map.
- Visualize the four relative thermal intensity classes.
- Show individual patch predictions and class scores.
- Communicate with the FastAPI backend.

The application should clearly identify the source scene and coordinate reference system. Geographic locations must be verified from the original scene metadata before assigning results to a specific city.

## Limitations

- The target labels are relative thermal intensity classes rather than calibrated LST measurements.
- The available input values and indices require appropriate radiometric preprocessing for scientific interpretation.
- The experiment uses one satellite scene, limiting conclusions about generalization.
- Test accuracy alone does not provide a complete picture of class-wise performance.
- Attention rollout is exploratory and is not sufficient as a standalone causal explanation.
- The scene identifier and geographic location must be verified before location-specific conclusions are drawn.

## Future Enhancements

- Use Landsat Collection 2 Level-2 Surface Temperature products or appropriately calibrated thermal data.
- Calculate spectral indices using calibrated surface reflectance.
- Verify the scene identifier, acquisition date, and geographic coordinates.
- Evaluate the model across multiple scenes, seasons, and cities.
- Add confusion matrices, per-class precision and recall, macro-F1, and balanced accuracy.
- Compare the ViT with CNNs and conventional machine learning baselines.
- Evaluate attention maps against perturbation-based and gradient-based explanations.
- Improve the web application with validated geospatial overlays and clear uncertainty reporting.

## Project Significance

This project demonstrates an end-to-end prototype that integrates remote sensing, multispectral feature engineering, deep learning, explainable AI, backend API development, and interactive geospatial visualization.

It provides a foundation for further research into interpretable satellite image analysis while highlighting the importance of calibrated data, valid reference labels, and independent evaluation.

## References

1. Zhou, D. et al. (2019). Satellite Remote Sensing of Surface Urban Heat Islands: Progress, Challenges, and Perspectives. *Remote Sensing, 11*(1), 48. https://doi.org/10.3390/rs11010048
2. Shi, H. et al. (2021). Urban heat island and its regional impacts using remotely sensed thermal data—A review of recent developments and methodology. *Land, 10*(8), 867. https://doi.org/10.3390/land10080867
3. Dosovitskiy, A. et al. (2021). An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale. *International Conference on Learning Representations (ICLR)*.
4. Jain, S., & Wallace, B. C. (2019). Attention is not Explanation. *Proceedings of NAACL-HLT*, 3543–3556. https://doi.org/10.18653/v1/N19-1357
5. U.S. Geological Survey. Landsat Collection 2 Level-1 Data. https://www.usgs.gov/landsat-missions/landsat-collection-2-level-1-data

---

**Project:** Explainable Vision Transformer for Urban Heat Island Detection Using Satellite Imagery

**Course:** Information Visualization — IBSA303L

**Submitted by:**
- P. Roshiga Jelin — 24MIA1103
- C. A. Sailasri — 24MIA1122
