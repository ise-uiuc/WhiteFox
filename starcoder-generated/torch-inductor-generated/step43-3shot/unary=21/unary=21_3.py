
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(4, 2, 1, dilation=2, padding=2)
        self.conv2a = torch.nn.Conv2d(2, 1, 3, stride=1, padding=1) # Change stride and padding from here
        self.conv2b = torch.nn.Conv2d(2, 1, 1, stride=1, padding=0) # Change stride and padding from here
    def forward(self, x):
        v1 = self.conv1(x) 
        v2a = self.conv2a(v1[:, 0:1, :, :]) # Use only the first channel
        v2b = self.conv2b(v1[:, 1:2, :, :]) # Use only the second channel
        v3 = torch.tanh(v2a + v2b)
        return v3
# Inputs to the model
x = torch.randn(1, 4, 3, 3)
