
class Model(torch.nn.Module):
    def __init__(self, min, max):
        super().__init__()
        self.convolution = torch.nn.Conv2d(1, 2, (3, 5), stride=1)
        self.activation = torch.nn.ReLU(inplace=False)
        self.min = min
        self.max = max
    def forward(self, x1):
        v1 = self.convolution(x1) 
        v2 = self.activation(v1)
        v3 = torch.clamp_min(v2, self.min)
        v4 = torch.clamp_max(v3, self.max)
        return v4
min = 0
max = 256
# Inputs to the model
x1 = torch.randn(8,1,32,64)
