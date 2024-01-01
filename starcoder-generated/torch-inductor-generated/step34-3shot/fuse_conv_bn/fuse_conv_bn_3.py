
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv1d(7, 7, 3)
        self.batchnorm3d = torch.nn.BatchNorm3d(7)
    def forward(self, x1):
        s = self.conv1(x1)
        y = self.batchnorm3d(s)
        return y
# Inputs to the model
x1 = torch.rand(1, 7, 11, 11)
