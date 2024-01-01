
class Model(torch.nn.Module):
    def __init__(self, out_channels):
        super().__init__()
        self.linear = torch.nn.Conv2d(5, out_channels, kernel_size=1, stride=1, padding=0)
 
    def forward(self, x1, x2):
        v1 = self.linear(x1)
        v2 = v1 + x2
        v3 = torch.relu(v2)
        return v3

# Initializing the model
m = Model(10)

# Inputs to the model
x1 = torch.randn(1, 5, 64, 64)
x2 = torch.randn(1, 10, 64, 64)
