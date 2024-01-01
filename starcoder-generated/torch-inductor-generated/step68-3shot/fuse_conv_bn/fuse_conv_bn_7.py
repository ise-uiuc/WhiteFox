
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv2d = nn.Conv2d(10, 20, 5, 1)
        self.conv1d = nn.Conv1d(1, 20, 5, 1)
    def forward(self, x):
        return self.conv1d(x)
# Inputs to the model
x = torch.randn(10, 1, 28, 28)
