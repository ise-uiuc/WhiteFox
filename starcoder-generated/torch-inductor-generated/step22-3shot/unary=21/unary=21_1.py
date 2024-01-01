
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=(1,))
        self.batchnorm2 = nn.BatchNorm1d(64)
    def forward(self, x0):
        x1 = self.conv1(x0)
        x2 = self.batchnorm2(x1)
        x3 = torch.tanh(x1)
        return x3
# Inputs to the model
x0 = torch.randn(1, 8, 256)
