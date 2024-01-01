
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 16, 3, stride=1, padding=1)
    def forward(self, x):
        y1 = self.conv1(x)
        z1 = torch.tanh(y1)
        return z1  
# Inputs to the model
x = torch.rand(1, 1, 47, 63)
