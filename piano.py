import numpy as np
import matplotlib.pyplot as plt
import matplotlib


class Piano(object):
    def __init__(self, mode=0):
        self.markers_list = [{'id': 203, 'name': 'top_left', 'corner_ind': 2, 'corners': None},
                             {'id': 204, 'name': 'top_right', 'corner_ind': 3, 'corners': None},
                             {'id': 205, 'name': 'bottom_right', 'corner_ind': 0, 'corners': None},
                             {'id': 206, 'name': 'bottom_left', 'corner_ind': 1, 'corners': None}]
        self.key_list = [{'note': 'C4',  'x': 0},
                         {'note': 'C#4', 'x': 0.75},
                         {'note': 'D4',  'x': 1},
                         {'note': 'D#4', 'x': 1.75},
                         {'note': 'E4',  'x': 2},
                         {'note': 'F4',  'x': 3},
                         {'note': 'F#4', 'x': 3.75},
                         {'note': 'G4',  'x': 4},
                         {'note': 'G#4', 'x': 4.75},
                         {'note': 'A4',  'x': 5},
                         {'note': 'A#4', 'x': 5.75},
                         {'note': 'B4',  'x': 6},
                         {'note': 'C5',  'x': 7},
                         {'note': 'C#5', 'x': 7.75},
                         {'note': 'D5',  'x': 8},
                         {'note': 'D#5', 'x': 8.75},
                         {'note': 'E5',  'x': 9},
                         {'note': 'F5',  'x': 10},
                         {'note': 'F#5', 'x': 10.75},
                         {'note': 'G5',  'x': 11},
                         {'note': 'G#5', 'x': 11.75},
                         {'note': 'A5',  'x': 12},
                         {'note': 'A#5', 'x': 12.75},
                         {'note': 'B5',  'x': 13}]
        if mode == 0:
            self.keys_color = self._generate_colormap(len(self.key_list))
        else:
            self.keys_color = [[0, 0, 255] for x in range(len(self.key_list))]
        self.keys_im_polygon_list = None
        self.num_white_keys = self._get_num_white_keys()
        self.markers_ids = self._get_markers_ids()
        self.markers_names = self._get_markers_names()

        # self.white_key_width = 0.5  # In size of the AruCo marker units which is printed
        # self.white_key_height = 2   # In size of the AruCo marker units which is printed
        # self.black_key_width = 0.25  # In size of the AruCo marker units which is printed
        # self.black_key_height = 1   # In size of the AruCo marker units which is printed

    def update_coordinates(self, corners, ids):
        """

        :param corners: AruCo markers corners
        :param ids: AruCo markers IDs
        :return:
        """
        ids = ids.flatten()

        # Filter out any marker id which is not related to the piano
        ids_relevant = [x for x in ids if x in self.markers_ids]

        if len(ids_relevant) != len(self.markers_list):
            # Don't update the corners if we didn't find them all
            print("Did not update piano corners")
            return

        # Add 4 corners of markers to our dictionary
        for i_tmp, id_tmp in enumerate(ids_relevant):
            id_input = np.flatnonzero(ids == id_tmp)[0]
            self.markers_list[self.markers_ids.index(id_tmp)]['corners'] = corners[id_input].copy()

        # Get specific piano board corner from the markers corners
        top_left_item = self.markers_list[self.markers_names.index('top_left')]
        top_right_item = self.markers_list[self.markers_names.index('top_right')]
        bottom_right_item = self.markers_list[self.markers_names.index('bottom_right')]
        bottom_left_item = self.markers_list[self.markers_names.index('bottom_left')]
        top_left_corner = top_left_item['corners'][0][top_left_item['corner_ind']].astype(float)
        top_right_corner = top_right_item['corners'][0][top_right_item['corner_ind']].astype(float)
        bottom_right_corner = bottom_right_item['corners'][0][bottom_right_item['corner_ind']].astype(float)
        bottom_left_corner = bottom_left_item['corners'][0][bottom_left_item['corner_ind']].astype(float)

        # Find image vectors in coordinates of the piano.
        # 1 unit is equal to 1 white key
        v_right_top = (top_right_corner-1 - top_left_corner+1) / float(self.num_white_keys)
        v_right_bot = (bottom_right_corner-1 - bottom_left_corner+1) / float(self.num_white_keys)
        v_right = (v_right_top + v_right_bot) / 2.0

        v_down_left = (bottom_left_corner - top_left_corner)
        v_down_right = (bottom_right_corner - top_right_corner)
        v_down = (v_down_left + v_down_right) / 2.0
        print([v_right, v_down])

        piano_origin = top_left_corner

        self.keys_im_polygon_list = []
        for key in self.key_list:
            if '#' in key['note']:
                w = 0.5
                h = 0.5
            else:
                w = 1.0
                h = 1.0
            c = np.array([[piano_origin + key['x'] * v_right],
                          [piano_origin + (key['x'] + w) * v_right],
                          [piano_origin + (key['x'] + w) * v_right + h * v_down],
                          [piano_origin + (key['x']) * v_right + h * v_down]])
            self.keys_im_polygon_list.append(c)

    def get_key_polygon(self, key_ind):
        return self.keys_im_polygon_list[key_ind].astype(np.int32)

    def get_key_color(self, key_ind):
        return self.keys_color[key_ind]

    def is_initialize(self):
        return self.keys_im_polygon_list is not None

    @staticmethod
    def _generate_colormap(num_of_levels):
        # define the piano keys colormap
        cmap = matplotlib.cm.get_cmap('brg')
        norm = matplotlib.colors.Normalize(vmin=0, vmax=num_of_levels)
        cmap_keys = [cmap(norm(i))[:3] for i in range(num_of_levels)]
        cmap_keys = [np.uint8(np.round(255 * np.array(x))).tolist() for x in cmap_keys]
        return cmap_keys

    def _get_markers_ids(self):
        return [x['id'] for x in self.markers_list]

    def _get_markers_names(self):
        return [x['name'] for x in self.markers_list]

    def _get_num_white_keys(self):
        return len([x for x in self.key_list if '#' not in x['note']])

    def get_key_index_by_name(self, name):
        tmp = [x['note'] for x in self.key_list]
        return tmp.index(name)
