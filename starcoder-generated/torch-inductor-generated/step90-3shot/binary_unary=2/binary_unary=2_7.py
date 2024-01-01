
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(28, 48, 5, stride=1, padding=0, groups=4)
    def forward(self, input1):
        v1 = self.conv(input1)
        v2 = v1 - 68
        v3 = F.relu(v2)
        v4 = torch.squeeze(v3, -1)
        return v4
# Inputs to the model
input1 = torch.rand(1, 28, 28, 28)
