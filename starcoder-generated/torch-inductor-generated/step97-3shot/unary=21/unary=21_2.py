
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 3, kernel_size=1, stride=1)
        self.conv2 = torch.nn.Conv1d(3, 17, kernel_size=(1,5), stride=1)
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = torch.tanh(x1)
        return self.conv2(x2)
# Inputs to the model
x = torch.randn(1, 1, 48, 71)
