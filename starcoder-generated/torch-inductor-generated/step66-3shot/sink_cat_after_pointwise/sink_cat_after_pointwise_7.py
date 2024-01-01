
class Model(torch.nn.Module):
    def __init__(self, x):
        super().__init__()
        self.conv1 = torch.nn.Conv1d(max(x, dim=1), 3, 3)
    def forward(self, x):
        x = torch.cat([x, x], dim=0)
        y = self.conv1(x)
        return y
# Inputs to the model
x = torch.randn(2, 3, 4)
