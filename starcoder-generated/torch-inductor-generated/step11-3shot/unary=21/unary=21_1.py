
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super(ModelTanh, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 32, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 16, 1, stride=1, padding=0)
    def forward(self, x):
        v1 = F.relu(self.conv1(x))
        v2 = torch.tanh(self.conv2(v1))
        return v2
# Inputs to the model
x = torch.randn(1, 1, 28, 28)
