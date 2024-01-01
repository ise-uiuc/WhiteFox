
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 1, 1, padding=0, stride=1, groups=1)
    def forward(self, x):
        v1 = self.conv(x)
        v2 = v1 - torch.sum(x, [1,2,3], keepdim=True)
        v3 = F.relu(v2)
        return v3
# Inputs to the model
x = torch.randn(1, 1, 42, 84, 28)
