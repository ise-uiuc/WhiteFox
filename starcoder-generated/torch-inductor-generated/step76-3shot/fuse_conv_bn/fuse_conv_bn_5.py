
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = torch.nn.ReLU(inplace=True)
        self.conv = torch.nn.Conv2d(3, 3, 3, stride=2, padding=1, bias=False)
    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x
# Inputs to the model
x = torch.randn(1, 3, 6, 6)
