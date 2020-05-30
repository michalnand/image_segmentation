import json
import numpy
from PIL import Image, ImageOps
import os
import torch

class SegmentationDataset:
    def __init__(self, json_file_name): 

        with open(json_file_name) as json_file:
            self.json_config = json.load(json_file)

        self.channels   = int(self.json_config["input_channels"])
        self.height     = int(self.json_config["input_height"])
        self.width      = int(self.json_config["input_width"])

        self.target_channels   = int(self.json_config["target_channels"])

        self.input_shape     = (self.channels, self.height, self.width)
        self.output_shape    = (self.target_channels, self.height, self.width)


        self.input_path     = str(self.json_config["input_path"])
        self.target_path    = str(self.json_config["target_path"])

        augmentations_count = int(self.json_config["augmentations_count"]) + 1

        input_files     = self._find_files(self.input_path)
        target_files    = self._find_files(self.target_path)

        self.count = len(input_files)


        self.input  = numpy.zeros((self.count*augmentations_count, self.channels,        self.height, self.width))
        self.target = numpy.zeros((self.count*augmentations_count, self.target_channels, self.height, self.width))


        for i in range(self.count):
            for augmentation in range(augmentations_count):
                if numpy.random.randint(2) == 0:
                    flip    = True
                else:
                    flip    = False

                if numpy.random.randint(2) == 0:
                    mirror  = True
                else:
                    mirror  = False

                crop_area_level = float(self.json_config["crop_area_level"])

                crop_area   = crop_area_level + (1.0 - crop_area_level)*numpy.random.rand()
                noise_level = numpy.random.rand()*float(self.json_config["noise_level"])


                if augmentation == 0:
                    flip        = False
                    mirror      = False
                    crop_area   = 1.0
                    noise_level = 0.0

                print("loading ", input_files[i], target_files[i])

                input_img    = self._load_image(input_files[i], True, flip, mirror, crop_area)
                target_img   = self._load_image(target_files[i], False, flip, mirror, crop_area)

                if self.channels == 3:
                    to_rgb = True
                else:
                    to_rgb = False

                input_np     = self._image_to_numpy(input_img, to_rgb, True, noise_level)

                target_np    = self._image_to_numpy(target_img, False, False)
                target_np    = numpy.clip(target_np, 0, self.target_channels)

            
                self.input[i]   = input_np.copy()
                self.target[i]  = target_np.copy()

    def get_count(self):
        return self.count

    def get_batch(self, batch_size = 32):
        input   = torch.zeros((batch_size, self.channels, self.height, self.width))
        target  = torch.zeros((batch_size, self.target_channels, self.height, self.width))

        for i in range(batch_size):
            idx = numpy.random.randint(self.count)
            input[i]  = torch.from_numpy(self.input[idx])
            target[i] = torch.from_numpy(self.target[idx])

        return input, target
        

    def _find_files(self, path):
        files_list = []

        for file in os.listdir(path):
            if file.endswith(".png"):
                files_list.append(str(file))

        result = []
        for file_name in files_list:
            result.append(path + file_name)

        result.sort()

        return result

    def _load_image(self, file_name, to_rgb, flip = False, mirror = False, crop_area = 1.0 ):
        image = Image.open(file_name)

        image = image.crop((image.width*(1 - crop_area), image.height*(1 - crop_area), image.width*crop_area, image.height*crop_area))

        image = image.resize((self.width, self.height), Image.LANCZOS)

        if flip:
            image = ImageOps.flip(image)

        if mirror:
            image = ImageOps.mirror(image)

        if to_rgb:
            image       = image.convert('RGB')
        else:
            image       = image.convert('L')

        return image

    def _image_to_numpy(self, image, to_rgb, normalize, noise_level = 0):

        if to_rgb:
            image       = image.convert('RGB')
            image_np    = numpy.array(image).astype(numpy.uint8)
            image_np    = numpy.rollaxis(image_np, 2, 0) 
        else:
            image       = image.convert('L')
            image_np    = numpy.array(image).astype(numpy.uint8)
            image_np    = numpy.expand_dims(image_np, axis = 0)
        
        if normalize:
            image_np    = image_np/255.0
            noise_mul   = 1
        else:
            noise_mul   = 255 

        if noise_level > 0:
            rnd      = noise_mul*noise_level*numpy.random.randn(*image_np.shape)
            image_np = (1.0 - noise_level)*image_np + noise_level*rnd

        if normalize:
            image_np = numpy.clip(image_np, 0, 1)
        else:
            image_np = numpy.clip(image_np, 0, 255)  

        return image_np      
