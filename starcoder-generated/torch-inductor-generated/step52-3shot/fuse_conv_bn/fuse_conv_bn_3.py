
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(100, 100)
        self.relu = torch.nn.ReLU(inplace=False)
    def forward(self, x3):
        y0 = self.linear(x3)
        y1 = self.relu(y0)
        y = y1 * y0
        return y1
# Inputs to the model
x3 = torch.randn(1, 100)
