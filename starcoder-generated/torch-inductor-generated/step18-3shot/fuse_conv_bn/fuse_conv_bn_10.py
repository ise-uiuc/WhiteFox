
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1d = torch.nn.Conv1d(5, 5, 6)
        self.conv2d = torch.nn.Conv1d(5, 5, 6)
        self.conv3d = torch.nn.Conv1d(5, 5, 6)
    def forward(self, x1):
        t = self.conv1d(x1)
        s = self.conv2d(x1)
        y = self.conv3d(x1)
        return t
# Inputs to the model
x1 = torch.randn(1, 5, 61)
