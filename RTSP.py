import cv2
import threading
import copy

class RTSCapture(cv2.VideoCapture):
    _cur_frame = None
    _reading = False
    schemes = ["rtsp://", "rtmp://"]

    @staticmethod
    def create(url, *schemes):
        rtscap = RTSCapture(url)
        rtscap.frame_receiver = threading.Thread(target=rtscap.recv_frame, daemon=True)
        rtscap.schemes.extend(schemes)
        if isinstance(url, str) and url.startswith(tuple(rtscap.schemes)):
            rtscap._reading = True
        elif isinstance(url, int):
            pass
        return rtscap

    def isStarted(self):
        ok = self.isOpened()
        if ok and self._reading:
            ok = self.frame_receiver.is_alive()
        return ok

    def recv_frame(self):
        while self._reading and self.isOpened():
            ok, frame = self.read()
            if not ok: break
            self._cur_frame = frame
        self._reading = False

    def read2(self):
        frame = self._cur_frame
        self._cur_frame = None
        return frame is not None, frame

    def start_read(self):
        self.frame_receiver.start()
        self.read_latest_frame = self.read2 if self._reading else self.read

    def stop_read(self):
        self._reading = False
        if self.frame_receiver.is_alive(): self.frame_receiver.join()

    def get_fps(self):
        return self.get(cv2.CAP_PROP_FPS)

# 处理RTSP流
rtsp_url = "rtsp://192.168.144.25:8554/main.264" #这里修改为说明书内官方rtsp地址！！！！
rtscap = RTSCapture.create(rtsp_url)
rtscap.start_read()

if rtscap.isStarted():
    print("RTSP stream opened successfully.")
else:
    print("Failed to open RTSP stream.")

fourcc = cv2.VideoWriter_fourcc(*'xvid')
fps = 8.0
width, height = 1920, 1080
out = cv2.VideoWriter('result1.avi', fourcc, fps, (width, height))

frame_count = 0
fps_all = 0

while rtscap.isStarted():
    t1 = cv2.getTickCount()
    frame_count += 1
    ok, frame = rtscap.read_latest_frame()
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break
    if not ok:
        continue

    # 深复制图像帧
    img0 = copy.deepcopy(frame)

    t2 = cv2.getTickCount()
    infer_time = (t2 - t1) / cv2.getTickFrequency()
    fps = 1.0 / infer_time
    fps_all += fps
    str_fps = f'fps:{fps:.4f}'
    cv2.putText(img0, str_fps, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    out.write(img0)
    cv2.imshow("cam", img0)

rtscap.stop_read()
rtscap.release()
out.release()
cv2.destroyAllWindows()

if frame_count > 0:
    print(f"All frames: {frame_count}, average fps: {fps_all / frame_count:.2f} fps")
else:
    print("No frames processed.")
