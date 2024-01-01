
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d_16 = torch.nn.Conv2d(4, 6, 4, stride=4, padding=1)
        self.relu_17 = torch.nn.ReLU()
    def forward(self, x35092):
        v1 = self.conv2d_16(x35092)
        v2 = self.relu_17(v1)
        return v2
# Inputs to the model
x35092 = torch.randn(238, 4, 32, 32)
