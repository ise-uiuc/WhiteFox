
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv1d(16, 32, 1)
        self.conv2 = torch.nn.Conv1d(32, 4, 1)
    def forward(self, x):
        f1 = torch.exp(-torch.sum(torch.abs(self.conv1(x)), dim=1))
        f2 = torch.abs(self.conv2(f1))
        f3 = self.conv2(f2)
        return f3
# Inputs to the model
x = torch.randn(1, 4, 10)
