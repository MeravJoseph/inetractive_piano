import numpy as np
import matplotlib.pyplot as plt
import matplotlib


class Piano(object):
    def __init__(self):
        self.num_piano_keys = 7 * 2
        self.markers_list = [{'id': 203, 'name': 'top_left', 'corner_ind': 2, 'corners': None},
                             {'id': 204, 'name': 'top_right', 'corner_ind': 3, 'corners': None},
                             {'id': 205, 'name': 'bottom_right', 'corner_ind': 0, 'corners': None},
                             {'id': 206, 'name': 'bottom_left', 'corner_ind': 1, 'corners': None}]

        self.white_key_width = 0.5  # In size of the AruCo marker units which is printed
        self.white_key_height = 2   # In size of the AruCo marker units which is printed
        self.black_key_width = 0.25  # In size of the AruCo marker units which is printed
        self.black_key_height = 1   # In size of the AruCo marker units which is printed
        self.keys_color = self._generate_colormap(self.num_piano_keys)
        self.keys_polygon_list = None

    def update_coordinates(self, corners, ids):
        """

        :param corners: AruCo markers corners
        :param ids: AruCo markers IDs
        :return:
        """
        ids = ids.flatten()

        # Filter out any marker id which is not related to the piano
        markers_ids = self._get_markers_ids()
        markers_names = self._get_markers_names()
        ids_relevant = [x for x in ids if x in markers_ids]

        if len(ids_relevant) != len(self.markers_list):
            # Don't update the corners if we didn't find them all
            print("Did not update piano corners")
            return

        for i_tmp, id_tmp in enumerate(ids_relevant):
            j = np.flatnonzero(id_tmp == markers_ids)
            self.markers_list[j[0]]['corners'] = corners[i_tmp]

        # Get specific piano board corner from the markers corners
        top_left_item = self.markers_list[markers_names.index('top_left')]
        top_right_item = self.markers_list[markers_names.index('top_right')]
        bottom_right_item = self.markers_list[markers_names.index('bottom_right')]
        bottom_left_item = self.markers_list[markers_names.index('bottom_left')]
        top_left_corner = top_left_item['corners'][0][top_left_item['corner_ind']]
        top_right_corner = top_right_item['corners'][0][top_right_item['corner_ind']]
        bottom_right_corner = bottom_right_item['corners'][0][bottom_right_item['corner_ind']]
        bottom_left_corner = bottom_left_item['corners'][0][bottom_left_item['corner_ind']]

        # Find image vectors in coordinates of the piano.
        # 1 unit is equal to 1 white key
        v_right_top = (top_right_corner - top_left_corner) / float(self.num_piano_keys)
        v_right_bot = (bottom_right_corner - bottom_left_corner) / float(self.num_piano_keys)
        v_right = (v_right_top + v_right_bot) / 2.0

        v_down_left = (bottom_left_corner - top_left_corner)
        v_down_right = (bottom_right_corner - top_right_corner)
        v_down = (v_down_left + v_down_right) / 2.0

        piano_origin = top_left_corner

        self.keys_polygon_list = []
        for key in range(self.num_piano_keys):
            c = np.array([[piano_origin + key * v_right],
                          [piano_origin + (key + 1) * v_right],
                          [piano_origin + (key + 1) * v_right + v_down],
                          [piano_origin + (key) * v_right + v_down]])
            self.keys_polygon_list.append(c)

    def get_key_polygon(self, key_ind):
        return self.keys_polygon_list[key_ind].astype(np.int32)

    def get_key_color(self, key_ind):
        return self.keys_color[key_ind]

    def is_initialize(self):
        return self.keys_polygon_list is not None

    @staticmethod
    def _generate_colormap(num_of_levels):
        # define the piano keys colormap
        cmap = matplotlib.cm.get_cmap('Set2')
        norm = matplotlib.colors.Normalize(vmin=0, vmax=num_of_levels)
        cmap_keys = [cmap(norm(i))[:3] for i in range(num_of_levels)]
        cmap_keys = [np.uint8(np.round(255 * np.array(x))).tolist() for x in cmap_keys]
        return cmap_keys

    def _get_markers_ids(self):
        return [x['id'] for x in self.markers_list]

    def _get_markers_names(self):
        return [x['name'] for x in self.markers_list]
