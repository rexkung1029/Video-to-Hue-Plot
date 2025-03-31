# Video to Hue Plot

本專案旨在分析影片（.mp4）或資料檔案（.npy）中每一幀的平均色相值（Average Hue），並透過互動式 Dash 應用程式展示與分析結果。分析過程包含影片逐幀處理、資料儲存及圖表生成，適合用於資料視覺化及後續研究分析。

---

## 目錄

- [簡介](#簡介)
- [功能](#功能)
- [環境需求](#環境需求)
- [安裝與執行](#安裝與執行)
- [授權](#授權)

---

## 簡介

本程式利用 Python 及相關套件（如 OpenCV、NumPy、Pandas、Plotly 與 Dash），進行影片影像處理，計算每一幀的平均色相值，並將分析結果以互動式圖表呈現。使用者可根據需求分析影片資料或載入現有的 .npy 檔案進行後續分析。

---

## 功能

- **影片處理與資料儲存**  
  讀取 .mp4 影片，逐幀計算平均色相值，並將結果儲存成 .npy 與 Excel 檔案，以便重複使用與資料交換。

- **互動式圖表生成**  
  利用 Plotly 與 Dash 建立動態互動圖表，使用者可選取特定資料點並生成選取點間隔分析圖，便於進一步探討數據特性。

- **多執行緒處理**  
  使用 ThreadPoolExecutor 進行並行運算，加快大量影片幀資料的處理速度，降低記憶體使用。

---

## 環境需求

- Python 3.6 以上版本  
- 套件：  
  - OpenCV (`opencv-python`)  
  - NumPy  
  - Pandas  
  - Plotly  
  - Dash  
  - tqdm

可參考 [Python 官方文件](https://docs.python.org/3/) 與各套件的官方文件了解更多細節。

---

## 安裝與執行

1. **安裝 Python 與相關套件**

   請先安裝 Python，然後使用 pip 安裝所有需求套件：

   ```bash
   pip install opencv-python numpy pandas plotly dash tqdm
   ```
2. **下載專案**

   從 GitHub 上 Clone 或下載此專案：
   ```bash
   git clonen https://github.com/rexkung1029/Video-to-Hue-Plot.git
   ```
3. **執行程式**

   使用命令列工具執行程式，並傳入欲分析的檔案路徑（支援 .mp4 或 .npy）：
   ```bash
   python hue_analysis.py input_path
   ```



## 授權

[MIT](https://choosealicense.com/licenses/mit/)
