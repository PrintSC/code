# ICSD-YOLO

The corresponding paper title for this project is “ICSD-YOLO: Intelligent Detection for Real-time Industrial Field Safety”. In the future, various data and codes in the paper will gradually be opened up

![](\assets\model_comparison.jpg)

![](assets\show.jpg)

# Object Detection

## 1、Requirements

We highly suggest using our provided dependencies to ensure reproducibility:

```
pip install requirements.txt
```

## 2、Train  your Net

```
yolo detect train data=cfg your data.yaml model=your model.yaml epochs=800 batch=128 imgsz=640 device=[0,1]
```

## 3、Main Results on  Industrial Field Dataset 

| **Models**  | **Input Size** | **FLOPs (G)** | **Params (M)** | **Top-1 (%)** | **Recall (%)** |
| :---------: | :------------: | :-----------: | :------------: | :-----------: | :------------: |
| ICSD-YOLO-N |    640x640     |      6.1      |      2.4       |     85.4      |      67.4      |
| ICSD-YOLO-S |    640x640     |     11.4      |      2.8       |     87.2      |      70.7      |
| ICSD-YOLO-M |    640x640     |     11.7      |      5.4       |     86.8      |      68.7      |
| ICSD-YOLO-L |    640x640     |     63.5      |      41.8      |     86.6      |      66.2      |
| ICSD-YOLO-X |    640x640     |     116.8     |      86.1      |     87.6      |      79.4      |

