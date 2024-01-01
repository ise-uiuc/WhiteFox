
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(64, 128, 1)
    def forward(self, x):
        x = self.conv1(x)
        x = torch.tanh(x)
        return x
# Inputs to the model
x = torch.randn(1,64,64,16)
