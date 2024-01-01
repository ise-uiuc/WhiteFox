
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(1, 1, kernel_size=3, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv(x1)
        return nn.Sigmoid()(v1)
# Inputs to the model
x1 = torch.randn(1, 10, 280)
