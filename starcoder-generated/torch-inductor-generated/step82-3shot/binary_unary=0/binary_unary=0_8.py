
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(512, 512, 3, stride=1, padding=0)
    def forward(self, x):
        v1 = self.conv(x)
        v2 = v1 + 0
        v3 = torch.relu(v2)
        return v3
# Inputs to the model
x = [ torch.randn(1, 512, 256, 256) ]
