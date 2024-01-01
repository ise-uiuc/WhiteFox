
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv1d(17, 35, 1, stride=1)
    def forward(self, x):
        x1 = torch.tanh(x)
        x2 = torch.tanh(x1)
        x3 = torch.tanh(x2)
        return self.conv1(x3)
# Inputs to the model
x = torch.randn(1, 17, 2)
