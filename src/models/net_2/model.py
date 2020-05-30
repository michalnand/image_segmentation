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
       

        self.encoder_0_layers = [ 
                                    nn.Conv2d(input_channels, 8, kernel_size=3, stride=2, padding=1),
                                    nn.ReLU(), 
                                    nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU()
                                ]

        self.encoder_1_layers = [
                                    nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
                                    nn.ReLU(), 
                                    nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU()
                                ]
        
        self.encoder_2_layers = [
                                    nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                                    nn.ReLU(), 
                                    nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU()
                                ]


        self.encoder_3_layers = [                        
                                    nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
                                    nn.ReLU(), 
                                    nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU()
                                ]
        
        self.decoder_layers =   [
                                    nn.Conv2d(16 + 32 + 64 + 64, 32, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(),
                                    nn.Conv2d(32, outputs_count, kernel_size=1, stride=1, padding=0)
                                ] 

        self.encoder_0 = nn.Sequential(*self.encoder_0_layers)
        self.encoder_0.to(self.device)

        self.encoder_1 = nn.Sequential(*self.encoder_1_layers)
        self.encoder_1.to(self.device)

        self.encoder_2 = nn.Sequential(*self.encoder_2_layers)
        self.encoder_2.to(self.device)

        self.encoder_3 = nn.Sequential(*self.encoder_3_layers)
        self.encoder_3.to(self.device)

        self.decoder = nn.Sequential(*self.decoder_layers)
        self.decoder.to(self.device)


        self.up_0 = nn.Upsample(scale_factor = 2, mode='nearest')
        self.up_1 = nn.Upsample(scale_factor = 4, mode='nearest')
        self.up_2 = nn.Upsample(scale_factor = 8, mode='nearest')
        self.up_3 = nn.Upsample(scale_factor = 16, mode='nearest')
        
        print(self.encoder_0)
        print(self.encoder_1)
        print(self.encoder_2)
        print(self.encoder_3)
        print(self.decoder)


    def forward(self, input):
        f0 = self.encoder_0(input)
        f1 = self.encoder_1(f0)
        f2 = self.encoder_2(f1)
        f3 = self.encoder_3(f2)

        f0_up = self.up_0(f0)
        f1_up = self.up_1(f1)
        f2_up = self.up_2(f2)
        f3_up = self.up_3(f3)

        decoder_input = torch.cat((f0_up, f1_up, f2_up, f3_up), 1)
 
        return self.decoder.forward(decoder_input)
   
    def save(self, path):
        torch.save(self.encoder_0.state_dict(), path + "trained/encoder_0.pt")
        torch.save(self.encoder_1.state_dict(), path + "trained/encoder_1.pt")
        torch.save(self.encoder_2.state_dict(), path + "trained/encoder_2.pt")
        torch.save(self.encoder_3.state_dict(), path + "trained/encoder_3.pt")
        torch.save(self.decoder.state_dict(), path + "trained/decoder.pt")

    def load(self, path):
        self.encoder_0.load_state_dict(torch.load(path + "trained/encoder_0.pt", map_location = self.device))
        self.encoder_0.eval() 

        self.encoder_1.load_state_dict(torch.load(path + "trained/encoder_1.pt", map_location = self.device))
        self.encoder_1.eval() 

        self.encoder_2.load_state_dict(torch.load(path + "trained/encoder_2.pt", map_location = self.device))
        self.encoder_2.eval() 

        self.encoder_3.load_state_dict(torch.load(path + "trained/encoder_3.pt", map_location = self.device))
        self.encoder_3.eval() 

        self.decoder.load_state_dict(torch.load(path + "trained/decoder.pt", map_location = self.device))
        self.decoder.eval() 


if __name__ == "__main__":
    input_shape = (1, 256, 256)
    outputs_count = 1

    model = Model(input_shape, outputs_count)

    input = torch.rand((4,) + input_shape)

    output = model.forward(input)

    print(output.shape)
