
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 3, 3)
        self.relu = torch.nn.ReLU(inplace=True)
        self.conv2 = torch.nn.Conv2d(3, 5, 3)
    def forward(self, x):
        v1 = self.conv1(x)
        v1 = self.relu(v1)
        v3 = self.conv2(v1)
        return v3
# Inputs to the model
x = torch.randn(1, 3, 28, 28)
