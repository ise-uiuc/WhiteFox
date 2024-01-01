
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(4, 128, 5, padding=2)
        self.relu = torch.nn.ReLU()
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = self.relu(v1)
        return v2
# Inputs to the model
x = torch.randn(2, 4, 224, 224)
