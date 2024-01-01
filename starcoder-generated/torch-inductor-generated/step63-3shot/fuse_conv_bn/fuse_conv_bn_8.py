
x2 = torch.randn(1, 4, 8, 8)

def get_layers(x):
    layers = []
    layers.append(torch.nn.ReLU())
    layers.append(torch.nn.BatchNorm2d(4))
    layers.append(torch.nn.functional.interpolate)
    layers.append(torch.nn.Conv2d(10, 5, 1, 1, 0, 1, 1, bias=False))

    layers.append(torch.nn.Conv2d(x[1], x[2], x[1], x[3], 1, x[2], x[2], bias=False))
    layers.append(torch.nn.BatchNorm2d(x[2]),)
    layers.append(torch.nn.MaxPool2d(3, stride=2, padding=1, dilation=1, ceil_mode=False, return_indices=False))
    out = layers[1](layers[4](layers[3](x)))
    return out

class Model(nn.Module):
    def __init__(self,):
        super().__init__() 
        
    def forward(self, x):
        out = get_layers(x)
        return out

# Inputs to the model
x = torch.randn(1, 7, 3, 3)
