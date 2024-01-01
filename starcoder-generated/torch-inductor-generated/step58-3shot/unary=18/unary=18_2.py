
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(3, 3, 3, stride=3, padding=29)
    def forward(self, x1):
        v1 = self.conv(x1)
        return torch.nn.Sigmoid()(v1)
# Inputs to the model
x1 = torch.randn(20, 3, 60)
