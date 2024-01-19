import cv2

from utils import Resolution

BLUE = (255, 0, 0)  # BGR
GREEN = (0, 255, 0)
LINE_THICKNESS = 5
FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
MAX_CAMERAS_ATTEMPTS = 3


class Camera:
    def __init__(self):
        self.camera = self.__access_camera()
        self.frame = None
        self._dimensions = None
        
    def __del__(self):
        self.camera.release()
    
    def __access_camera(self):
        for i in range(MAX_CAMERAS_ATTEMPTS):
            camera = cv2.VideoCapture(0)
            if camera.read()[0]:
                return camera
    
    @property        
    def dimensions(self):
        if self._dimensions is None:
            camera_width = int(self.camera.get(3))
            camera_height = int(self.camera.get(4))
            self._dimensions = Resolution(camera_width, camera_height)

        return self._dimensions
            
    def update_frame(self):
        ret, frame = self.camera.read()

        if not ret:
            print("Lost camera feed")
        self.frame = frame
        
    def track_eye(self):
        # TODO: currently does not deal with multiple eyes/faces
        
        greyscale_frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)

        # consider center of face as eyes due to avoid more error-prone eye cascade
        faces = FACE_CASCADE.detectMultiScale(greyscale_frame, 1.3, 5)

        # Parallax effect can only be seen by 1 person at a time so only take the
        # first set of eyes seen
        if len(faces) == 0:
            return None
        
        # x, y, w, h = faces[0]
        return faces[0]
    
    def draw_eye_tracking(self, face):
        drawn_frame = self.frame.copy()
        eye = face
        if eye is not None:
            x, y, w, h = eye

            cv2.rectangle(drawn_frame, (x, y), (x + w, y + h), BLUE, LINE_THICKNESS)

            cv2.circle(drawn_frame, (x + int(w/2), y + int(h/2)), 5, GREEN, -1)

        cv2.imshow("Eye Tracking", drawn_frame)
        