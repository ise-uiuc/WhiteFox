
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 3, padding=1)
        self.conv2 = torch.nn.Conv1d(64, 64, 3, dilation=2, padding=4)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x
# Inputs to the model
x = torch.randn(8, 3, 224)
