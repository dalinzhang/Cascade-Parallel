import numpy as np
import tensorflow as tf

print("****************************************")
print("number of parameters to be trained:")
print(np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))
total_parameters = 0
for variable in tf.trainable_variables():
    # shape is an array of tf.Dimension
    shape = variable.get_shape()
    print("shape: ",shape)
    print("length of the shape: ",len(shape))
    variable_parameters = 1
    for dim in shape:
        print(dim)
        variable_parameters *= dim.value
    print("number of parameters of the shape: ",variable_parameters, "\n")
    total_parameters += variable_parameters
print(total_parameters)
print("****************************************")
