import cv2

class VideoStream:
    def __init__(self, src=0, width=640, height=480):
        self.cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        if width and height:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def read(self):
        ok, frame = self.cap.read()
        return ok, frame

    def release(self):
        if self.cap.isOpened():
            self.cap.release()

if __name__ == "__main__":
    vs = VideoStream()
    while True:
        ok, frame = vs.read()
        if not ok:
            break
        cv2.imshow("Sanity Check - Webcam", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break
    vs.release()
    cv2.destroyAllWindows()
