
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = torch.relu
    def forward(self, x):
        y = torch.cat([x, x, x], dim=2)
        y = y.view(y.size()[0], -1)
        y = self.relu(y)
        return y
# Inputs to the model
x = torch.randn(2, 3, 4)
