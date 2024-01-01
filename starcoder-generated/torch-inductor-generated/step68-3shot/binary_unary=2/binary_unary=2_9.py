
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(nn.Conv2d(6, 1, 1, stride=1, padding=0),
        nn.Flatten())
    def forward(self, x1):
        v1 = self.seq(x1)
        v2 = v1 - 0.7
        v3 = F.relu(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 6, 32, 32)
