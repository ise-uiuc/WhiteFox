
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1  = torch.nn.Conv2d(in_channels=7, out_channels=8, kernel_size=3, stride=2)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = torch.relu(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 7, 244, 244)
