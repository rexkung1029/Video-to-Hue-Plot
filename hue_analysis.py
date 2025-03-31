import os
import cv2
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import plotly.graph_objs as go
import dash
from dash import dcc, html, Input, Output, State

def calculate_average_hue(frame):
    """計算單一幀的平均色相值 (Average Hue)"""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    return np.mean(hue)

def frame_generator(cap, resize_width=None):
    """生成器，逐幀讀取並調整解析度"""
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if resize_width:
            height, width = frame.shape[:2]
            aspect_ratio = height / width
            new_height = int(resize_width * aspect_ratio)
            frame = cv2.resize(frame, (resize_width, new_height))
        yield frame

def process_video(video_path, max_workers=32, max_pending_futures=100):
    """
    處理影片，計算每幀的平均色相值，並顯示進度條。
    使用 ThreadPoolExecutor 並限制同時進行的任務數量以減少記憶體使用。
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"無法開啟影片檔案: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    average_hues = []
    timestamps = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        frame_idx = 0

        with tqdm(total=total_frames, desc="Processing frames") as pbar:
            for frame in frame_generator(cap, resize_width=640):
                future = executor.submit(calculate_average_hue, frame)
                futures[future] = frame_idx
                frame_idx += 1

                if len(futures) >= max_pending_futures:
                    for completed_future in as_completed(list(futures.keys())):
                        idx = futures.pop(completed_future)
                        try:
                            avg_hue = completed_future.result()
                            average_hues.append(avg_hue)
                            timestamps.append(idx / fps)
                        except Exception as e:
                            print(f"處理幀 {idx} 時出錯: {e}")
                        pbar.update(1)
                        if len(futures) < max_pending_futures:
                            break

            for future in as_completed(futures):
                idx = futures[future]
                try:
                    avg_hue = future.result()
                    average_hues.append(avg_hue)
                    timestamps.append(idx / fps)
                except Exception as e:
                    print(f"處理幀 {idx} 時出錯: {e}")
                pbar.update(1)

    cap.release()

    sorted_data = sorted(zip(timestamps, average_hues))
    
    sorted_data_array = np.array(sorted_data)
    np.save('hue_data.npy', sorted_data_array)

    sorted_timestamps, sorted_average_hues = zip(*sorted_data) if sorted_data else ([], [])

    df = pd.DataFrame({
        'Timestamp (s)': sorted_timestamps,
        'Average Hue': sorted_average_hues
    })

    return df, fps, duration

def create_initial_plot(df, selected_points):
    """創建初始的 Plotly 圖表，並根據選取點設置標記顏色和大小"""
    fig = go.Figure()

    fig.add_trace(go.Scattergl(
        x=df['Timestamp (s)'],
        y=df['Average Hue'],
        mode='lines',
        name='Average Hue',
        line=dict(color='blue'),
        hoverinfo='x+y'
    ))

    if selected_points:
        selected_df = pd.DataFrame(selected_points, columns=['Timestamp (s)', 'Average Hue'])
        fig.add_trace(go.Scattergl(
            x=selected_df['Timestamp (s)'],
            y=selected_df['Average Hue'],
            mode='markers',
            name='Selected Points',
            marker=dict(size=10, color='red'),
            hoverinfo='x+y'
        ))

    fig.update_layout(
        title='Average Hue Over Time',
        xaxis_title='Time (s)',
        yaxis_title='Average Hue',
        dragmode='pan',
    )
    return fig

def create_interval_plot(selected_df):
    """創建選取時間點的間隔 Plotly 圖表"""
    if selected_df.empty:
        return go.Figure()
    intervals = np.diff(selected_df['Timestamp (s)'])
    average_intervals = np.average(intervals) if intervals.size > 0 else 0
    fig = go.Figure()
    fig.add_trace(go.Scattergl(
        x=selected_df['Timestamp (s)'][1:],
        y=intervals,
        mode='markers+lines',
        name='Selected Points',
        marker=dict(size=10, color='red')
    ))
    fig.update_layout(
        title=f'Selected Time Points Average Intervals \n Average Interval {average_intervals:.2f}s',
        xaxis_title='Time (s)',
        yaxis_title='Interval Time'
    )
    return fig

def save_html(fig, path):
    """將 Plotly 圖表儲存為 HTML"""
    fig.write_html(path)

def save_excel(df, path):
    """將 DataFrame 儲存為 Excel 檔案"""
    df.to_excel(path, index=False)

def get_output_paths(file_path):
    """
    根據輸入檔案路徑生成輸出檔案的路徑。
    不論輸入為影片（.mp4）或資料（.npy），皆依據檔名進行命名。
    """
    directory = os.path.dirname(file_path)
    initial_plot_path = os.path.join(directory, f"hue_plot.html")
    interval_plot_path = os.path.join(directory, f"selected_plot.html")
    excel_path = os.path.join(directory, f"selected.xlsx")
    return initial_plot_path, interval_plot_path, excel_path

def run_dash_app(df, initial_plot, output_paths):
    """啟動 Dash 應用程式，處理用戶互動"""
    app = dash.Dash(__name__)
    app.layout = html.Div([
        html.H1("Average Hue Analysis"),
        dcc.Store(id='selected-points-store', data=[]),
        dcc.Store(id='relayout-store', data={}),
        dcc.Graph(
            id='initial-graph',
            figure=initial_plot,
            config={'scrollZoom': True}
        ),
        html.Button("生成選取間隔圖表和匯出 Excel", id='generate-button', n_clicks=0, style={'marginTop': '20px'}),
        html.Button("取消選取所有點", id='clear-button', n_clicks=0, style={'marginTop': '20px', 'marginLeft': '10px'}),
        dcc.Graph(
            id='interval-graph'
        ),
        html.Div(id='output-message', style={'marginTop': 20})
    ])

    @app.callback(
        Output('relayout-store', 'data'),
        Input('initial-graph', 'relayoutData'),
        State('relayout-store', 'data')
    )
    def store_relayout(relayoutData, stored_relayout):
        """
        儲存用戶的視圖狀態，只儲存 xaxis.range 與 yaxis.range
        """
        if relayoutData is not None:
            if 'xaxis.range' in relayoutData:
                stored_relayout['xaxis.range'] = relayoutData['xaxis.range']
            elif 'xaxis.range[0]' in relayoutData and 'xaxis.range[1]' in relayoutData:
                stored_relayout['xaxis.range'] = [relayoutData['xaxis.range[0]'], relayoutData['xaxis.range[1]']]
            
            if 'yaxis.range' in relayoutData:
                stored_relayout['yaxis.range'] = relayoutData['yaxis.range']
            elif 'yaxis.range[0]' in relayoutData and 'yaxis.range[1]' in relayoutData:
                stored_relayout['yaxis.range'] = [relayoutData['yaxis.range[0]'], relayoutData['yaxis.range[1]']]
        return stored_relayout

    @app.callback(
        [
            Output('initial-graph', 'figure'),
            Output('interval-graph', 'figure'),
            Output('output-message', 'children'),
            Output('selected-points-store', 'data')
        ],
        [
            Input('initial-graph', 'clickData'),
            Input('generate-button', 'n_clicks'),
            Input('clear-button', 'n_clicks')
        ],
        [
            State('selected-points-store', 'data'),
            State('relayout-store', 'data')
        ]
    )
    def handle_interactions(clickData, generate_n_clicks, clear_n_clicks, selected_data, stored_relayout):
        ctx = dash.callback_context

        if not ctx.triggered:
            return dash.no_update, dash.no_update, dash.no_update, selected_data

        trigger = ctx.triggered[0]['prop_id'].split('.')[0]
        message = ""
        interval_fig = dash.no_update
        updated_fig = dash.no_update

        if trigger == 'initial-graph' and clickData:
            point = clickData['points'][0]
            timestamp = point['x']
            hue = point['y']
            if [timestamp, hue] not in selected_data:
                selected_data.append([timestamp, hue])
                message = f"已選取 {len(selected_data)} 個點。"
            else:
                selected_data = [p for p in selected_data if not (p[0] == timestamp and p[1] == hue)]
                message = f"已取消選取。剩餘 {len(selected_data)} 個點。"

            updated_fig = create_initial_plot(df, selected_data)
            if 'xaxis.range' in stored_relayout:
                updated_fig.update_xaxes(range=stored_relayout['xaxis.range'])
            if 'yaxis.range' in stored_relayout:
                updated_fig.update_yaxes(range=stored_relayout['yaxis.range'])
            return updated_fig, dash.no_update, message, selected_data

        elif trigger == 'generate-button' and generate_n_clicks > 0:
            if not selected_data:
                message = "沒有選取任何點。"
                return dash.no_update, dash.no_update, message, selected_data
            selected_data = sorted(selected_data)
            selected_data_array = np.array(selected_data)
            np.save('selected_data.npy', selected_data_array)

            selected_df = pd.DataFrame(selected_data, columns=['Timestamp (s)', 'Average Hue'])
            interval_fig = create_interval_plot(selected_df)

            initial_plot_path, interval_plot_path, excel_path = output_paths

            excel_df = pd.DataFrame([pt[0] for pt in selected_data], columns=['Timestamp (s)'])

            save_html(create_initial_plot(df, selected_data), initial_plot_path)
            save_html(interval_fig, interval_plot_path)
            save_excel(excel_df, excel_path)
            print("Excel 輸出路徑：", excel_path)
            message = f"已儲存圖表和 Excel 至 {os.path.dirname(excel_path)}"
            return dash.no_update, interval_fig, message, selected_data
        
        elif trigger == 'clear-button' and clear_n_clicks > 0:
            selected_data = []
            message = "已取消所有選取的點。"
            updated_fig = create_initial_plot(df, selected_data)
            if 'xaxis.range' in stored_relayout:
                updated_fig.update_xaxes(range=stored_relayout['xaxis.range'])
            if 'yaxis.range' in stored_relayout:
                updated_fig.update_yaxes(range=stored_relayout['yaxis.range'])
            interval_fig = create_interval_plot(pd.DataFrame())
            return updated_fig, interval_fig, message, selected_data

        return dash.no_update, dash.no_update, dash.no_update, selected_data

    app.run_server(debug=False)

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="計算影片的平均色相值並生成圖表與 Excel 檔案。"
    )
    parser.add_argument('input_path', type=str, help='請輸入影片檔案（.mp4）或資料檔案（.npy）的路徑')
    parser.add_argument('--nodash', action='store_true', help='只獲取 sorted_data.npy，不啟動 Dash 程式')
    args = parser.parse_args()

    input_path = args.input_path
    if not os.path.isfile(input_path):
        print(f"檔案不存在: {input_path}")
        return

    file_ext = os.path.splitext(input_path)[1].lower()

    if file_ext == '.mp4':
        print("處理影片中，請稍候...")
        df, fps, duration = process_video(input_path)
    elif file_ext == '.npy':
        print("載入已儲存的 .npy 資料檔案...")
        sorted_data_array = np.load(input_path)
        if sorted_data_array.size == 0:
            print("讀取到空的資料檔案。")
            return
        sorted_timestamps, sorted_average_hues = zip(*sorted_data_array)
        df = pd.DataFrame({
            'Timestamp (s)': sorted_timestamps,
            'Average Hue': sorted_average_hues
        })
        fps = None
        duration = sorted_timestamps[-1] if sorted_timestamps else 0
    else:
        print("不支援的檔案格式，請提供 .mp4 或 .npy 檔案")
        return

    if args.nodash:
        print("啟動選項為 nodash，不啟動 Dash 程式。")
        sorted_data_path = os.path.join(os.path.dirname(input_path), "sorted_data.npy")
        print("sorted_data.npy 已產生，路徑：", sorted_data_path)
        return

    print("生成圖表...")
    initial_fig = create_initial_plot(df, [])
    output_paths = get_output_paths(input_path)

    print("啟動 Dash 應用程式，請在瀏覽器中進行互動選取。")
    run_dash_app(df, initial_fig, output_paths)

if __name__ == "__main__":
    main()