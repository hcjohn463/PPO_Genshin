import threading
import queue
import time
from collections import deque
from ultralytics import YOLO
class BackgroundObjectDetector:
    def __init__(self, model_path="best.pt", queue_size=2):
        self.model = YOLO(model_path)
        self.frame_queue = queue.Queue(maxsize=queue_size)
        self.result_queue = queue.Queue(maxsize=queue_size)
        self.latest_result = None
        self.running = True
        self.detection_thread = threading.Thread(target=self._detection_loop, daemon=True)
        self.detection_thread.start()
    
    def _detection_loop(self):
        while self.running:
            try:
                # 非阻塞方式獲取最新frame
                try:
                    # 清空舊的frame
                    while self.frame_queue.qsize() > 1:
                        self.frame_queue.get_nowait()
                    frame = self.frame_queue.get_nowait()
                except queue.Empty:
                    time.sleep(0.01)  # 短暫休眠避免CPU過度使用
                    continue

                # 執行檢測
                results = self.model.predict(source=frame, conf=0.7)
                
                # 處理結果
                boss_position = None
                for result in results[0]:
                    if result.boxes.xyxy.shape[0] > 0:
                        boxes = result.boxes.xyxy
                        boss_position = [
                            boxes[0, 0].item(),
                            boxes[0, 1].item(),
                            boxes[0, 2].item(),
                            boxes[0, 3].item(),
                            0
                        ]
                        break

                # 更新最新結果
                self.latest_result = boss_position
                
                # 清空結果隊列並添加新結果
                while not self.result_queue.empty():
                    self.result_queue.get_nowait()
                self.result_queue.put_nowait(boss_position)
                
            except Exception as e:
                print(f"Detection error: {e}")
                time.sleep(0.1)

    def add_frame(self, frame):
        """非阻塞方式添加新幀"""
        try:
            if self.frame_queue.qsize() < self.frame_queue.maxsize:
                self.frame_queue.put_nowait(frame)
        except queue.Full:
            pass  # 如果隊列滿了，跳過這一幀
    
    def get_latest_result(self):
        """非阻塞方式獲取最新結果"""
        return self.latest_result
    
    def stop(self):
        """停止檢測線程"""
        self.running = False
        if self.detection_thread.is_alive():
            self.detection_thread.join()
# 在主程式中使用：
if __name__ == '__main__':
    # 你現有的初始化代碼...
    EPISODES = 5000
    # 創建檢測器實例
    detector = BackgroundObjectDetector()
    
    for episode in range(EPISODES):
        # 你的episode初始化代碼...
        
        while True:
            frame = get_screen(screen_area)
            detector.add_frame(frame)
            boss_position = detector.get_latest_result()
            
            if boss_position is not None:
                block = adjust_view(boss_position, block)
            
            # 其餘的遊戲邏輯保持不變...
            
    # 清理
    detector.stop()