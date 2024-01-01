
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.c2d = torch.nn.Conv2d(16, 16, (2, 2))
        self.conv = torch.nn.Sequential(torch.nn.Conv1d(7, 7, 2), torch.nn.BatchNorm1d(7))
    def forward(self, input1):
        return self.c2d(input1)
# Inputs to the model
input1 = torch.randn(1, 16, 8)
