
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1d=torch.nn.Conv1d(3, 3, padding=(0,1))
    def forward(self, x):
        negative_slope = 0.5058561
        v1 = self.conv1d(x)
        v2 = v1 > 0
        v3 = v1 * negative_slope
        v4 = torch.where(v2, v1, v3)
        return v4
# Inputs to the model
x1=torch.randn(1, 3, 1, 1)
