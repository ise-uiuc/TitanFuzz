def set_memory_growth():
    import tensorflow as tf

    physical_devices = tf.config.list_physical_devices("GPU")
    try:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
    except:
        # Invalid device or cannot modify virtual devices once initialized.
        pass


set_memory_growth()
