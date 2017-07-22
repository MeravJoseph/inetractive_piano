import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import cv2
from media import Display, calibrate_projector

PARAMS = {}
PARAMS['aruco'] = {'dict': cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL),
                   'marker_size_cm': 45, 'markers_dist_cm': 18}
PARAMS['camera'] = {'cam_mtx': np.load("/home/merav/code/proj1/calibration/cam_mtx.npy"),
                    'dist_coeffs': np.load("/home/merav/code/proj1/calibration/dist_coeffs.npy")}
PARAMS['piano'] = {'white_key_width_cm': 2.3, 'white_key_height_cm': 7.8,
                   'black_key_width_cm': 1.2, 'black_key_height_cm': 4.5,
                   'left_marker_tr_to_piano_cm': np.array([1.1, 1.8]),
                   'size_cm': np.array([15.5, 7.7])}
SCREEN_SIZE = (1366, 768)  # (width, height)
IMG_SIZE = (640, 480)   # (width, height)
NUM_OF_PIANO_KEYS = 7 * 2

piano_key_ind = 0

# define the piano keys colormap
cmap = matplotlib.cm.get_cmap('Set2')
norm = matplotlib.colors.Normalize(vmin=0, vmax=NUM_OF_PIANO_KEYS)
cmap_keys = [cmap(norm(i))[:3] for i in range(NUM_OF_PIANO_KEYS)]
cmap_keys = [np.uint8(np.round(255*np.array(x))).tolist() for x in cmap_keys]

if __name__ == "__main__":
    # Calibrate camera and projector - find homography
    cam_to_proj = calibrate_projector(screen_size=SCREEN_SIZE, aruco_dict=PARAMS['aruco']['dict'])

    # Start projector
    display = Display()

    # Start camera
    cap = cv2.VideoCapture(1)
    while True:
        # Capture frame-by-frame
        ret, img = cap.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imshow('camera', img)
        
        # Test camera-to-projector transformation by projecting camera image
        # back through projector
        if False:
            # Transform image to projector for debug
            dst = cv2.warpPerspective(img, cam_to_proj, SCREEN_SIZE)
            display.show_array(dst)

            cv2.imshow('img', img)
            # Wait for key from user
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break
            continue

        # Find the piano board AruCo markers
        corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray, PARAMS['aruco']['dict'])
        cv2.aruco.drawDetectedMarkers(img, corners, ids)

        # If no markers were found continue to next frame
        if ids is None:
            continue

        if len(ids) > 0:
            ids = ids.flatten()
            # corners [marker_num][0][coord_num][axis]
            id_tl = np.flatnonzero(ids == 203)
            id_br = np.flatnonzero(ids == 205)
            if len(id_tl) > 0:
                corners_tl = corners[id_tl[0]][0]
            elif len(id_br) > 0:
                corners_br = corners[id_br[0]][0]
            else:
                continue

            # Build piano keys map according to marker location and size
            # Each item in list hold the corners of each piano key, in pixels units in camera coordinates
            white_key_width = 0.5  # In size of AruCo marker units
            white_key_height = 2  # In size of AruCo marker units
            v_right = (corners_tl[1] - corners_tl[0]) * white_key_width  # vector in right direction. size of white key
            v_down = (corners_tl[2] - corners_tl[1]) * white_key_height  # vector in down direction. size of white key
            piano_origin = corners_tl[2]
            piano_keys_corners = []
            for key in range(NUM_OF_PIANO_KEYS):
                c = np.array([[piano_origin + key * v_right],
                              [piano_origin + (key + 1) * v_right],
                              [piano_origin + (key + 1) * v_right + v_down],
                              [piano_origin + (key) * v_right + v_down]])
                piano_keys_corners.append(c)

        # Select key to project
        img_to_project = np.zeros((IMG_SIZE[1], IMG_SIZE[0], 3), np.uint8)
        pts = piano_keys_corners[piano_key_ind].astype(np.int32)
        color = cmap_keys[piano_key_ind]
        cv2.fillPoly(img_to_project, [pts], color, cv2.LINE_AA)
        cv2.putText(img_to_project, "%d" % piano_key_ind, tuple(pts[3, 0, :]),
                    cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 0, 0))

        # Transform image to projector coordinates
        dst = cv2.warpPerspective(img_to_project, cam_to_proj, SCREEN_SIZE)
        dst = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
        display.show_array(dst)

        # for c in corners:
        #     rvec, tvec = cv2.aruco.estimatePoseSingleMarkers(c, 0.01*PARAMS['aruco']['marker_size_cm'],
        #                                                      PARAMS['camera']['cam_mtx'],
        #                                                      PARAMS['camera']['dist_coeffs'])
        #     cv2.aruco.drawAxis(img, PARAMS['camera']['cam_mtx'], PARAMS['camera']['dist_coeffs'],
        #                        rvec, tvec, 0.1)

        # Draw and display the markers
        cv2.imshow('camera', img)
        cv2.imshow('img_to_project', img_to_project)

        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break
        for n in range(NUM_OF_PIANO_KEYS):
            if n < 10:
                if key & 0xFF == ord('%d' % n):
                    piano_key_ind = n

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

    display.close()
