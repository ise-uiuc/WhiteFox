
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(16, 2, 1, stride=1, padding=0)
        self.relu = torch.nn.ReLU()
    def forward(self, x56):
        v1 = self.conv(x56)
        v2 = self.relu(v1)
        return v2
# Inputs to the model
x56 = torch.randn(1, 16, 22, 43)
