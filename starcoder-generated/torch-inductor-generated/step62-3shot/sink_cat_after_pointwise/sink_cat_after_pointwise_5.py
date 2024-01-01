
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.y = torch.randn(1, 2)
    def forward(self, x):
        y = torch.cat((x, self.y), dim=1)
        y = torch.relu(y)
        return y
# Inputs to the model
x = torch.randn(2, 1)
