
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv1d(32, 64, 7, stride=1, padding=3)
    def forward(self, x2):
        r1 = self.conv(x2)
        r2 = torch.tanh(r1)
        return r2
# Inputs to the model
x2 = torch.randn(10, 32,230)
