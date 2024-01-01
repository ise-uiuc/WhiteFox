
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 16, 3, 1, 1, bias=False)
    def forward(self, t):
        e1 = self.conv(t)
        e2 = e1 - e1
        v3 = e2.squeeze(0)
        v4 = F.relu(v3)
        return F.relu(v4)
# Inputs to the model
x0 = torch.randn(1, 1, 32, 32)
