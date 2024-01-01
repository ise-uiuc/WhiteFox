
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 16, 1, stride=1, padding=1)
 
        # The bias vector for the convolution layer
        self.bias = torch.nn.Parameter(torch.zeros(16))
 
    def forward(self, x1, min_value, max_value):
        x2 = F.relu(self.conv(x1) + self.bias)
 
        # Clamp the output of convolution to a minimum value of 0
        x3 = x2.clamp_min_(0)
 
        # Clamp the output of convolution to a maximum value of 1
        x4 = x3.clamp_max_(1)
 
        return x4
 
# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 96, 96)
