
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = torch.nn.Conv2d(8, 16, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(16, 32, 3, stride=1, padding=1)
    def forward(self, input):
        x1 = F.relu(self.conv1(input))
        x2 = F.relu(self.conv2(input))
        x3 = x1 - x2
        x4 = torch.squeeze(x3, 0)
        return x4
# Inputs to the model
x = torch.randn(1, 8, 44, 44)
