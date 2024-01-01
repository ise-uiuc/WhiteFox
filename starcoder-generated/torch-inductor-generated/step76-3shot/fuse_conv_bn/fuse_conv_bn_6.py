
class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(8, 8, 3)
        self.relu = torch.nn.ReLU(inplace=False)
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        return x
# Inputs to the model
x = torch.randn(8, 8, 4, 4)
