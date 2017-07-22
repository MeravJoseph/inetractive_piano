from __future__ import print_function
import os
import numpy as np
import pygame
import time
import cv2


def calibrate_projector(screen_size, aruco_dict):
    """ Calibrate projector to camera.
        Finds the Homography between the coordinate systems.
    :param screen_size: (x,y). Should be (1366, 768)
    :param aruco_dict: OpenCV's AruCo dictionary type. From: cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
    :return:
    """
    # Create grid board with AruCo markers
    board = cv2.aruco.GridBoard_create(9, 6, 0.025, .0125, aruco_dict)
    img_board = board.draw((screen_size[0]-200, screen_size[1]-60))
    img_board = np.pad(img_board, pad_width=10, mode='constant', constant_values=255)
    img_board_rgb = np.dstack((img_board, img_board, img_board))

    # Detect markers on projector image
    corners_proj, ids_proj, rejectedImgPoints = cv2.aruco.detectMarkers(img_board, aruco_dict)
    ids_proj = ids_proj.flatten()
    # cv2.aruco.drawDetectedMarkers(img_board_rgb, corners_proj, ids_proj)
    # plt.figure(), plt.imshow(img_board_rgb)
    # plt.show()

    # Project markers
    display = Display()
    display.show_array(img_board_rgb)

    # Detect markers using the camera
    proj_to_cam = None
    cap = cv2.VideoCapture(1)
    while True:
        # Capture frame-by-frame
        ret, img = cap.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the marker corners
        corners_cam, ids_cam, rejectedImgPoints = cv2.aruco.detectMarkers(gray, aruco_dict)

        # If no markers were found continue to next frame
        if ids_cam is None:
            continue

        ids_cam = ids_cam.flatten()

        # Draw markers on RGB image and display it
        cv2.aruco.drawDetectedMarkers(img, corners_cam, ids_cam)
        cv2.putText(img, "Press 'c' to calibrate", (10, 30), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0))
        cv2.imshow('img', img)

        # Wait for key from user
        key = cv2.waitKey(10)
        if key & 0xFF == ord('q'):
            break
        if key & 0xFF == ord('c'):
            # Build a list of all markers corners detected both in camera and projector image
            ids_intersection = list(set(ids_proj).intersection(set(ids_cam)))
            proj_points = []
            cam_points = []
            for id in ids_intersection:
                c_proj = corners_proj[np.flatnonzero(ids_proj == id)[0]][0]
                proj_points.append(c_proj)
                c_cam = corners_cam[np.flatnonzero(ids_cam == id)[0]][0]
                cam_points.append(c_cam)
            proj_points = np.vstack(proj_points)
            cam_points = np.vstack(cam_points)

            # Find homography
            cam_to_proj, _ = cv2.findHomography(cam_points, proj_points)
            print("Camera to projector:")
            print(cam_to_proj)
            break
    display.close()
    return cam_to_proj


class Display:
    def __init__(self, screen_width=1366):
        """

        :param screen_width: The width of the left display (laptop), so that the code will know
            where the second right monitor (projector) starts.
        """
        self.screen_x = screen_width
        self.screen_y = 0
        os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (self.screen_x, self.screen_y)
        self.open()

    def show_array(self, array):
        a = np.swapaxes(array.copy(), 0, 1)
        surf = pygame.surfarray.make_surface(a)
        #surf = pygame.transform.scale(surf, self.size)
        self.screen.blit(surf, (0, 0))
        pygame.display.flip()

    def show_image(self, image):
        image = self.scale_image(image)
        self.screen.blit(image, (0, 0))
        pygame.display.flip()

    def scale_image(self, image):
        return pygame.transform.scale(pygame.image.load(image), self.size)

    def open(self):
        pygame.display.init()
        #pygame.mouse.set_visible(False)
        self.screen = pygame.display.set_mode((self.screen_x, self.screen_y), pygame.NOFRAME)
        screen_size = self.screen.get_size()
        self.size = screen_size
        # print(self.size)

    def close(self):
        pygame.display.quit()


class Sound(object):
    def __init__(self, wav_directory):
        self.wav_directory = wav_directory
        pygame.mixer.init()
        self.wav_dict = self._load_wav_files()

    def play_note_sound(self, name):
        """ Plays note sound

        :param name: Note name, like in Piano class: "C#4", "D5", ...
        """
        self.wav_dict[name].play()

    def _load_wav_files(self):
        """
        Loads the note .wav files into memory.
        :return: dictionary
        """
        files = os.listdir(self.wav_directory)
        d = dict()
        for f in files:
            fn, ext = os.path.splitext(f)
            if not (ext == ".wav"):
                continue
            d[fn] = pygame.mixer.Sound(os.path.join(self.wav_directory, f))
        return d



if __name__ == "__main__":
    display = Display()
    import numpy as np
    ind = 0
    import cv2
    im = cv2.imread("/home/merav/code/proj1/board_images/board_0000.png", cv2.IMREAD_COLOR)
    while True:
        im[100:300, :, ind % 3] = 255
        #display.show_image("/home/merav/code/proj1/board_images/board_%04d.png" % ind)
        display.show_array(im)
        time.sleep(1)
        ind += 1
        if ind > 6:
            break
    display.close()
