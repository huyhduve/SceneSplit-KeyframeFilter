# 🎬 SceneSplit-KeyframeFilter

> **Automatic scene segmentation and keyframe extraction pipeline**  
> Built with **TransNet V2**, **OpenCV**, **Histogram/SSIM filtering**

---

## 🧠 Overview

`SceneSplit-KeyframeFilter` is a modular computer-vision pipeline that:
1. **Detects scenes** in videos using **TransNet V2** (deep learning model for shot boundary detection).  
2. **Cuts frames** within each detected scene via **OpenCV**.  
3. **Filters redundant frames** using **Histogram**, **SSIM**

It’s designed for:
- Video summarization  
- Dataset preprocessing for VBS
- Scene-based highlight extraction  
- Research on video segmentation and representation learning  

---

## 🚀 Features

- 🎞️ **Scene segmentation** — robust shot detection with TransNet V2  
- 🧩 **Frame extraction** — fast OpenCV-based frame sampling per scene  
- 🧠 **Filtering & deduplication** — Histogram + SSIM similarity metrics  
- 💡 **Notebook-friendly** — visualize every step interactively  
- 💾 **Timestamp export** — saves `<start_frame> <end_frame>` for each scene  
--- 


