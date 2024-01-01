
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 32, 3, padding=(1, 1), bias=False)
        self.conv2 = torch.nn.Conv2d(32, 1, 1, bias=False)
    def forward(self, x):
        v1 = self.conv1(x)
        v2 =  torch.tanh(v1)
        v3 = self.conv2(v2)
        return v3
# Inputs to the model
x = torch.randn(1, 1, 64, 64)
