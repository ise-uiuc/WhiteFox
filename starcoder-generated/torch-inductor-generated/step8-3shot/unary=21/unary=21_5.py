
class tanhActivation(torch.nn.Module):
    # Defining the forward pass
    def forward(self, x):
        result = torch.tanh(x)
        return result
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 1, 3, stride=1, padding=1)
        self.tanh = tanhActivation()
    def forward(self, x):
        v1 = self.conv(x)
        v2 = self.tanh(v1)
        return v2.detach()
# Inputs to the model
x = torch.randn(64, 3, 64, 64)
