#encoding: utf-8
from keras.utils import multi_gpu_model

def train_on_data_parallel(CPU_model, optimizer, loss_funcs, metrics,
                           generator, steps, epochs, callback, 
                           n_works=5, val_gen=None, val_steps=None,
                           gpus=1, init_epoch=0):
    """note that model must instantiated in cpu

    Example:   
    ```python
        with tf.device('/cpu:0'):
            model = get_model(config)
        model = train_on_data_parallel(model, optimizer, loss, metrics,
                data_gen, steps, epochs)
        model.save_weights('myweights.h5')
    ```
    """
    parallel_model = multi_gpu_model(CPU_model, gpus=gpus)
    parallel_model.compile(loss=loss_funcs,
                           metrics=metrics,
                           optimizer=optimizer)
    parallel_model.fit_generator(generator, steps_per_epoch=steps,
                                 epochs=epochs, callbacks=callback,
                                 validation_data=val_gen,
                                 workers=n_works,
                                 use_multiprocessing=True,
                                 validation_steps=val_steps,
                                 initial_epoch=init_epoch)
    return CPU_model