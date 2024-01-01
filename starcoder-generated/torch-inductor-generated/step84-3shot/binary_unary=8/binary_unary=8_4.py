
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(5, 1, 5, stride=(1, 3)) # The 2nd dimension of the 1st input tensor is not 3
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.conv(x1)
        v3 = v1 + v2
        v4 = torch.relu(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 5, 28, 78)
