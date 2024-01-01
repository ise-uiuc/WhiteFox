
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(48, 96, 2, stride=1, padding=0)
        self.relu = torch.nn.ReLU(inplace=False)
    def forward(self, x19):
        v1 = self.conv1(x19)
        v2 = self.relu(v1)
        return v2
# Inputs to the model
x19 = torch.randn(1, 48, 48, 47)
