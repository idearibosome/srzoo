# SRZoo Model Conversion

It is possible to convert your own super-resolution models by using the provided model conversion tools.
Here are the instructions of converting the pre-trained models.


## Common

- Copy the conversion tools in this folder to your folder that contains your own codes.
- The conversion tools may generate the converted model named as ```model.pb```.


## TensorFlow

- Please specify name of the placeholder of the input image to ```sr_input```. For example, if your input placeholder is written as ```tf_input = tf.placeholder(...)```, then specify the name as:
```python
tf_input = tf.placeholder(tf.float32, [None, None, None, 3], name='sr_input')
```
- Use the following code snippet to convert the model.
```python
import tensorflow_converter

# Build the graph and restore the pre-trained variables.
# ...

tensorflow_converter.convert_to_srzoo(sess=sess, output=model.output)
```

Arguments of ```convert_to_srzoo```:
- ```sess```: The TensorFlow Session that contains the graph.
- ```output```: The Tensor that corresponds to the super-resolved output.


## Keras (with TensorFlow backend)

- Please specify name of the input image layer to ```sr_input```. For example, if your input layer is written as ```input_layer = keras.layers.InputLayer(...)```, then specify the name as:
```python
input_layer = keras.layers.InputLayer(input_shape=(None, None, 3), name='sr_input')
```
- Use the following code snippet to convert the model.
```python
import keras_converter

# Build and restore the model.
# ...

keras_converter.convert_to_srzoo(model=model)
```

Arguments of ```convert_to_srzoo```:
- ```model```: The Keras Model of the super-resolution.


## PyTorch

Converting PyTorch-based models require the 'pytorch2keras' module.
We provide the [forked version of pytorch2keras](https://github.com/idearibosome/pytorch2keras-srzoo), which is modified and optimized for SRZoo.

- Download and unarchive the ```pytorch2keras``` folder of the [forked version of pytorch2keras](https://github.com/idearibosome/pytorch2keras-srzoo).
- Use the following code snippet to convert the model.
```python
import pytorch_converter

# Build and restore the PyTorch-based model.
# ...

pytorch_converter.convert_to_srzoo(model=model)

```

Arguments of ```convert_to_srzoo```:
- ```model```: The PyTorch Model of the super-resolution.

