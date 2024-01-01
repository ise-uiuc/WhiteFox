
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1d_1 = torch.nn.Conv1d(5, 10, (4,), (2,))
        self.softsign_1 = torch.nn.Softsign()
    def forward(self, x1):
        x2 = self.conv1d_1(x1)
        x3 = self.softsign_1(x2)
        x4 = x1 * x3
        return x4
# Inputs to the model
x1 = torch.randn(1, 5, 500)
