"""
Loads a Teachable Machine image model (keras_model.h5) WITHOUT relying on
Keras' fragile model deserialization. It rebuilds the MobileNetV2 + dense head
and copies the weights straight out of the .h5 file, so it works on any
recent TensorFlow / Keras version.
"""
import h5py
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers


def _get(grp, *names):
    for n in names:
        if n in grp:
            return np.array(grp[n])
    raise KeyError(f"None of {names} found in {list(grp.keys())}")


def load_teachable_machine(path):
    with h5py.File(path, "r") as f:
        w = f["model_weights"]
        base_w, head_w = w["sequential_1"], w["sequential_3"]

        base = keras.applications.MobileNetV2(
            input_shape=(224, 224, 3), alpha=0.35, include_top=False, weights=None)

        for layer in base.layers:
            if not layer.weights:
                continue
            g = base_w[layer.name]
            if isinstance(layer, layers.BatchNormalization):
                vals = [_get(g, "gamma:0"), _get(g, "beta:0"),
                        _get(g, "moving_mean:0"), _get(g, "moving_variance:0")]
            elif isinstance(layer, layers.DepthwiseConv2D):
                vals = [_get(g, "depthwise_kernel:0", "kernel:0")]
            elif isinstance(layer, layers.Conv2D):
                vals = [_get(g, "kernel:0")]
            else:
                raise RuntimeError(f"Unexpected layer type: {layer}")
            layer.set_weights(vals)

        d1 = layers.Dense(100, activation="relu", name="dense_Dense1")
        d2 = layers.Dense(head_w["dense_Dense2"]["kernel:0"].shape[1],
                          activation="softmax", use_bias=False, name="dense_Dense2")
        inp = keras.Input((224, 224, 3))
        x = layers.GlobalAveragePooling2D()(base(inp))
        out = d2(d1(x))
        model = keras.Model(inp, out)
        d1.set_weights([_get(head_w["dense_Dense1"], "kernel:0"),
                        _get(head_w["dense_Dense1"], "bias:0")])
        d2.set_weights([_get(head_w["dense_Dense2"], "kernel:0")])
    return model
