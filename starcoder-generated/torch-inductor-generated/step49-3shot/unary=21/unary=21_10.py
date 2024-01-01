
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1d = torch.nn.Conv1d(3, 3, 1)
        self.tanh = torch.nn.Tanh()

    def forward(self, x):
        x1 = self.conv1d(x)
        x2 = self.tanh(x1)
        return x2
# Inputs to the model
x = torch.randn(1, 3, 7)
