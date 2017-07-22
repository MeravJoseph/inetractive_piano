import time
import numpy as np
import cv2
from media import Display, Sound, calibrate_projector
from piano import Piano


class Manager(object):
    def __init__(self):
        self.screen_size = (1366, 768)  # (width, height)
        self.camera_size = (640, 480)   # (width, height)
        self.key_quit = 'q'
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
        self.update_piano_corners_freq = 5  # Number of frames to update piano corners
        self.img_to_project = np.zeros((self.camera_size[1], self.camera_size[0], 3), np.uint8)
        # self.display = Display(screen_width=self.screen_size[0])
        self.piano = Piano()
        self.cam_to_proj = None  # Transformation from camera to projector coordinates
        self.song = ["G5", "E5", "E5", "br", "F5", "D5", "D5", "br", "C5", "D5", "E5", "F5", "G5", "G5", "G5",
         "br","G5", "E5", "E5", "br", "F5", "D5", "D5", "br", "C5", "E5", "G5", "G5", "C5"]

    def calibrate_cam_to_proj(self):
        self.cam_to_proj = calibrate_projector(screen_size=self.screen_size, aruco_dict=self.aruco_dict)

    def run(self):
        self.calibrate_cam_to_proj()
        display = Display(screen_width=self.screen_size[0])
        sound = Sound(wav_directory="wav")
        cap = cv2.VideoCapture(1)
        frame_num = 0
        note_num = 0    # note index in the song
        while True:
            # Get an image from camera
            img = self._get_image(cap_obj=cap)
            frame_num += 1
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Find the piano board AruCo markers
            corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray, self.aruco_dict)
            cv2.aruco.drawDetectedMarkers(img, corners, ids)

            # Display image for debug
            cv2.imshow('camera', img)

            # If no markers were found continue to next frame
            if ids is None:
                continue

            # If we found the piano markers
            if len(ids) > 0:
                if frame_num % self.update_piano_corners_freq == 0:
                    self.piano.update_coordinates(corners, ids)

            # Project a key
            self.img_to_project.fill(0)
            if self.piano.is_initialize():
                if self.song[note_num] != 'br':
                    # If note is not break
                    piano_key_ind = self.piano.get_key_index_by_name(self.song[note_num])
                    pts = self.piano.get_key_polygon(piano_key_ind)
                    color = self.piano.get_key_color(piano_key_ind)

                    cv2.fillPoly(self.img_to_project, [pts], color, cv2.LINE_AA)
                    cv2.putText(self.img_to_project, "%d" % piano_key_ind, tuple(pts[3, 0, :]),
                                cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 0, 0))

                    # Play note sound
                    sound.play_note_sound(self.piano.key_list[piano_key_ind]['note'])
                time.sleep(0.5)
                note_num += 1

            cv2.imshow('img_to_project', self.img_to_project)
            # Transform image to projector coordinates
            dst = cv2.warpPerspective(self.img_to_project, self.cam_to_proj, self.screen_size)
            dst = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
            display.show_array(dst)

            # Wait for key from user
            key = cv2.waitKey(1)
            if key & 0xFF == ord(self.key_quit):
                break
        # When everything done, release the capture
        cap.release()
        cv2.destroyAllWindows()
        display.close()

    def _get_image(self, cap_obj, debug_transformation=False):
        """ Get image from camera

        :param cap_obj: OpenCV's VideoCapture object
        :return: Image as NumPy array
        """
        # Capture frame-by-frame
        ret, img = cap_obj.read()

        # Test camera-to-projector transformation by projecting camera image
        # back through projector
        if debug_transformation:
            # Transform image to projector for debug
            dst = cv2.warpPerspective(img, self.cam_to_proj, self.screen_size)
            self.display.show_array(dst)
            cv2.imshow('debug', img)
        return img

if __name__ == "__main__":
    manager = Manager()
    manager.run()
