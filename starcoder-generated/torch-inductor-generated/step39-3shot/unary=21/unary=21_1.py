
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1d = torch.nn.Conv1d(40, 120, 2, stride=2, padding=1)
    def forward(self, x20):
        v21 = self.conv1d(x20)
        v22 = torch.tanh(v21)
        return v22
# Inputs to the model
x20 = torch.randn(3, 40, 14)
