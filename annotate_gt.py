import napari
import numpy as np
from magicgui.widgets import Container, PushButton, ComboBox
from napari.layers.labels import Labels

class Annotations(Container):

    def __init__(self, viewer: 'napari.Viewer'):
        super().__init__()
        self.viewer = viewer

        self._label_picker = ComboBox(label='Labels layer', choices=self._get_label_layers)

        self._new_cnt = None
        self._new_label_btn = PushButton(text='New Label')
        self._new_label_btn.clicked.connect(self._get_new_label)

        self.extend([self._label_picker, self._new_label_btn])

    def _get_label_layers(self, choices):
        choices = [layer for layer in self.viewer.layers if isinstance(layer, Labels)]
        return choices

    def _get_new_label(self, event):
        selected_layer_name = self._label_picker.current_choice
        selected_layer = self.viewer.layers[selected_layer_name]

        if self._new_cnt == None:
            current_max = np.max(selected_layer.data)
            self._og_max = current_max + 100
            self._new_cnt = 0
        
        selected_layer.selected_label = self._og_max + self._new_cnt
        self._new_cnt += 1
            
        

if __name__ == '__main__':
    viewer = napari.Viewer()
    widg = Annotations(viewer)
    viewer.window.add_dock_widget(widg)

    
    napari.run()
