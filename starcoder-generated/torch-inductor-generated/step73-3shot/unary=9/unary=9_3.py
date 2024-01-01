
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv = torch.nn.Conv2d(3, 1, 3)
    def forward(self, x1):
        x1 = torch.relu(x1)
        x4 = torch.add(x1, 3)
        x5 = torch.relu6(x4)
        x6 = torch.div(x5, 6)
        return x6, torch.tanh(x1) + x6
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
