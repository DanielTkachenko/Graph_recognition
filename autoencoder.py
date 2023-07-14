import keras.callbacks
from keras.models import Model
from keras.layers import Input, Dense, Flatten, Reshape, Conv2D,\
    Conv2DTranspose, MaxPooling2D, UpSampling2D, BatchNormalization
from keras.activations import relu

def get_arch(input_shape, levels, filters_min = 4,
             strides_shape_encoder=(1, 1), strides_shape_decoder = (1, 1),
             conv2d_encoder = True,
             conv2d_decoder = True,
             batch_normalization=False,
             max_pooling = True,
             up_sampling = True,
             conv2d_transpose = False,
             c2dt_strides_shape = (1, 1)):
    x = Input(shape=input_shape)
    input_img = x
    x = build_encoder(x, levels, filters_min, strides_shape_encoder,
                      conv2d_encoder, batch_normalization, max_pooling)
    x = build_decoder(x, levels, filters_min, strides_shape_decoder, conv2d_decoder, batch_normalization,
                      up_sampling, conv2d_transpose, c2dt_strides_shape)

    output = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)
    model = Model(input_img, output)
    model.compile(optimizer='adam', loss='mse')
    return model

def build_encoder(input, levels, filters_min, strides_shape, conv2d, batch_normalization, max_pooling, filters_num_down):
    for i in range(levels):
        if filters_num_down:
            filters_num = filters_min * (levels - i)
        else:
            filters_num = filters_min * (i + 1)
        if conv2d == True:
            input = Conv2D(filters_num, (3, 3),strides=strides_shape, padding='same')(input)
        if batch_normalization == True:
            input = BatchNormalization()(input)
        input = relu(input)
        if max_pooling == True:
            input = MaxPooling2D((2, 2), padding='same')(input)
    return input

def build_decoder(input, levels, filters_min, strides_shape, conv2d,
                  batch_normalization, up_sampling, conv2d_transpose, c2dt_strides_shape, filters_num_up):
    for i in range(levels):
        if filters_num_up:
            filters_num = filters_min * (i + 1)
        else:
            filters_num = filters_min * (levels - i)
        if conv2d == True:
            input = Conv2D(filters_num, (3, 3),strides=strides_shape, padding='same')(input)
        if conv2d_transpose == True:
            input = Conv2DTranspose(filters_min * (i + 1), (3, 3), strides=c2dt_strides_shape, padding='same')(input)
        if batch_normalization == True:
            input = BatchNormalization()(input)
        input = relu(input)
        if up_sampling == True:
            input = UpSampling2D((2, 2))(input)
    return input

def save_model(model, name):
    model.save(name)

def load_model(name):
    model = keras.models.load_model(name)
    return model