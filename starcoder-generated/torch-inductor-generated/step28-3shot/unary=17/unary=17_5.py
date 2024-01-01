
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 5, (2, 3), padding=(1, 1), stride=(9, 3), weight_attr=torch.nn.Parameter(torch.randn(48, 3, 2, 3)), bias_attr=True)
        self.conv2 = torch.nn.Conv2d(3, 5, 1, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 3, 32, 32)
