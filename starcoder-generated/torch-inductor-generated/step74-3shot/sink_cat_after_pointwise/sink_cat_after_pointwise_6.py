
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv1d(2, 3, 2, groups=2)
    def forward(self, x):
        y = torch.cat((x, y), dim=0)
        return self.conv(y)
# Inputs to the model
x = torch.randn(2, 3, 4)
