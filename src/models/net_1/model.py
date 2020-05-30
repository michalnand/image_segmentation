import torch
import torch.nn as nn

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class Model(torch.nn.Module):
    def __init__(self, input_shape, outputs_count):
        super(Model, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.input_shape    = input_shape
        self.outputs_count  = outputs_count
        
        input_channels  = self.input_shape[0]
       

        self.encoder_layers = [ 
                                nn.Conv2d(input_channels, 8, kernel_size=3, stride=2, padding=1),
                                nn.ReLU(), 
                                nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(),

                                nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),
                                nn.ReLU(), 
                                nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(),

                                nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
                                nn.ReLU(), 
                                nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(),

                                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                                nn.ReLU(), 
                                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(), 

                                nn.Upsample(scale_factor=16, mode='nearest')
                            ]
        
        self.decoder_layers = [
                                nn.Conv2d(64 + input_channels, 32, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(), 
                                nn.Conv2d(32, outputs_count, kernel_size=1, stride=1, padding=0)
                            ]

        for i in range(len(self.encoder_layers)):
            if hasattr(self.encoder_layers[i], "weight"):
                torch.nn.init.xavier_uniform_(self.encoder_layers[i].weight)

        for i in range(len(self.decoder_layers)):
            if hasattr(self.decoder_layers[i], "weight"):
                torch.nn.init.xavier_uniform_(self.decoder_layers[i].weight)

        self.encoder = nn.Sequential(*self.encoder_layers)
        self.encoder.to(self.device)

        self.decoder = nn.Sequential(*self.decoder_layers)
        self.decoder.to(self.device)

        print(self.encoder)
        print(self.decoder)


    def forward(self, input):
        features = self.encoder(input)
        decoder_input = torch.cat((features, input), 1)

        return self.decoder.forward(decoder_input)
   
    def save(self, path):
        torch.save(self.encoder.state_dict(), path + "trained/encoder.pt")
        torch.save(self.decoder.state_dict(), path + "trained/decoder.pt")

    def load(self, path):
        self.encoder.load_state_dict(torch.load(path + "trained/encoder.pt", map_location = self.device))
        self.encoder.eval() 

        self.decoder.load_state_dict(torch.load(path + "trained/decoder.pt", map_location = self.device))
        self.decoder.eval() 


if __name__ == "__main__":
    input_shape = (1, 256, 256)
    outputs_count = 1

    model = Model(input_shape, outputs_count)

    input = torch.rand((4,) + input_shape)

    output = model.forward(input)

    print(output.shape)
