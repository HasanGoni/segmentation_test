# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/03_viz_utils.ipynb.

# %% auto 0
__all__ = ['show_model']

# %% ../nbs/03_viz_utils.ipynb 3
from tensorflow.keras.utils import plot_model

# %% ../nbs/03_viz_utils.ipynb 4
def show_model(model, show_shapes=True, show_layer_names=True, to_file='model.png'):
    "Plot a Keras model as a graph."
    return plot_model(model, show_shapes=show_shapes, show_layer_names=show_layer_names, to_file=to_file)