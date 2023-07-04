import gradio as gr
import cv2
import numpy as np
from tensorflow.python.keras.models import load_model
from util import set_gpu, my_custom_2


def upload_image(image):
    processed_image = cv2.resize(image, (256, 256))
    processed_image = np.expand_dims(processed_image, axis=0)

    predicted_image = (loaded_model.predict(processed_image, batch_size=1, verbose=1))[0]
    predicted_image = np.clip(predicted_image, 0, 255).astype(np.uint8)

    return predicted_image


set_gpu()
my_model = 'my_model.h5'
loaded_model = load_model(my_model, custom_objects={'my_custom_2': my_custom_2})

title = 'Fisheye Image Rectification'
description = 'Ref: https://github.com/byunghyun23/GSAFE'
image_input = gr.inputs.Image(label="Upload an image", type='numpy')
output_image = gr.outputs.Image(label="Processed Image", type='numpy')
custom_css = '#component-12 {display: none;} #component-1 {display: flex; justify-content: center; align-items: center;} img.svelte-ms5bsk {width: unset;}'

iface = gr.Interface(fn=upload_image, inputs=image_input, outputs=output_image,
                     title=title, description=description, css=custom_css)
iface.launch(server_port=8080)