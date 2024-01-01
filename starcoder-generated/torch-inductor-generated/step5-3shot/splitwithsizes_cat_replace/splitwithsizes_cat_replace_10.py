
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.split0 = torch.nn.Conv2d(3, 1, 3, stride=1, padding=1)
        self.split1 = torch.nn.Conv2d(3, 1, 3, stride=1, padding=1)
        self.split2 = torch.nn.Conv2d(1, 1, 3, stride=1, padding=1)
 
    def forward(self, input_tensor, split_size):
        split = torch.split(input_tensor, split_size)
        out = split[0] + self.split0(split[1])
        out = out + self.split1(split[2]) + self.split2(split[3])
        out = torch.cat(out)
        return out

# Initializing the model
m = Model()

# Split sizes used in the model
split_size = torch.tensor([9, 9, 9, 9])
