
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = torch.nn.ReLU()
        self.conv = torch.nn.Conv2d(54, 3, 1, stride=1, bias=True, padding=0)
    def forward(self, x1):
        v1 = self.conv(self.relu(x1))
        return v1
# Inputs to the model
x1 = torch.randn(1, 54, 1, 1)
