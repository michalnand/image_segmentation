from segmentation_dataset import *

from models.net_2.model import Model

import numpy

from PIL import Image, ImageDraw


def show(input, output):

    width       = input.shape[1]
    height      = input.shape[0]
    input_rgb   = numpy.array([input, input, input])

    mask = numpy.ones(output.shape)*0.1 < output

    k = 0.5
    input_rgb[0] = (1.0 - k)*input_rgb[0]
    input_rgb[1] = (1.0 - k)*input_rgb[1]  + k*mask
    input_rgb[2] = (1.0 - k)*input_rgb[2]

    result = input_rgb
    result = numpy.rollaxis(result, 0, 2)
    result = numpy.rollaxis(result, 2, 1)

    result = (result*255).astype(dtype=numpy.uint8)
    image = Image.fromarray(result)
    image.show()



testing_dataset  = SegmentationDataset("dataset_config_testing.json")

model = Model(testing_dataset.input_shape, testing_dataset.output_shape[0])

model.load("models/net_2/")

batch_size = 5



input, target = testing_dataset.get_batch()

prediction = model.forward(input)


for i in range(batch_size):
    input_np        = input[i][0].detach().numpy()
    
    target_np       = numpy.clip(target[i][0].detach().numpy(), 0, 1)
    prediction_np   = numpy.clip(prediction[i][0].detach().numpy(), 0, 1)

    target_np       = numpy.round(target_np, 3)
    prediction_np   = numpy.round(prediction_np, 3)

    print("target = ")
    print(target_np, "\n")
    print("target = ")
    print(prediction_np, "\n")
    print("\n\n\n\n")
 
    show(input_np, prediction_np)