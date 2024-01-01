
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv5 = torch.nn.Conv2d(3, 64, (3, 3), stride=(1, 1), padding=(1, 1))
    def forward(self, x1):
        v1 = F.relu(x1)
        v2 = self.conv5(v1)
        v3 = v2 - 1.1
        v4 = F.relu(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 3, 127, 127)
