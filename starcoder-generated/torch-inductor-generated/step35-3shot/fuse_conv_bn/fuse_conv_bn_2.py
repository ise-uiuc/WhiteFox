
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv = torch.nn.Conv1d(3, 3, 3)
    def forward(self, x):
        x, y = self.conv(x), self.conv(x*2)
        return (x*y).relu()
# Inputs to the model
x = torch.randn(1, 3, 8)
