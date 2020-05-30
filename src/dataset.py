from segmentation_dataset import *

class Dataset:
    def __init__(self, config_json_training, config_json_testing):

        self.training = SegmentationDataset(config_json_training)
        self.testing  = SegmentationDataset(config_json_testing)


if __name__ == "__main__":
    dataset = Dataset("dataset_config_training.json", "dataset_config_testing.json")
    print("program done")
