
class Test(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, 3, stride=1, padding=1)
        self.relu = torch.nn.ReLU()
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        return x
# Inputs to the model
x = torch.randn(1, 3, 224, 244)
