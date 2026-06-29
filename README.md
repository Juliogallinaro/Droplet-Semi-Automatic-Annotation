# Deep Learning for Microfluidic Droplet Characterization via Semi-Automatic Annotation
Semi-automatic annotation pipeline for microfluidic droplets using template matching (OpenCV) to generate YOLO-format datasets for deep learning training.

## Train with Docker

```powershell
docker run --rm --gpus all `
	--shm-size=16g `
	-v "G:\dataset_final:/dataset" `
	-v "F:\CBEB26\runs:/runs" `
	ultralytics/ultralytics:8.4.82 `
	yolo train `
		model=yolov8n.pt `
		data=/dataset/dataset.yaml `
		epochs=100 `
		imgsz=640 `
		project=/runs `
		name=droplets_v1 `
		workers=4 `
		batch=8 `
		patience=10
```
