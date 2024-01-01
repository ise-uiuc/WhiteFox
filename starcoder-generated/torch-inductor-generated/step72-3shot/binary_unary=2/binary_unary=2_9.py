
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 3, 1, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = v1 - 1
        v3 = F.relu(v2)
        v4 = torch.transpose(v3, 0, 1).contiguous()
        return v4
# Inputs to the model
x1 = torch.randn(1, 1, 28, 28)
