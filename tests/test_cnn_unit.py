import pytest


def test_unfreeze_top_keeps_batchnorm_frozen():
    pytest.importorskip("tensorflow")
    import keras

    from soilclf.cnn import build_model, unfreeze_top

    _, base = build_model(3, 64, 0.3, weights=None)
    unfreeze_top(base, 30)
    top = base.layers[-30:]
    assert all(not l.trainable for l in base.layers[:-30])
    assert all(not l.trainable for l in top if isinstance(l, keras.layers.BatchNormalization))
    assert any(l.trainable for l in top if l.weights and not isinstance(l, keras.layers.BatchNormalization))
