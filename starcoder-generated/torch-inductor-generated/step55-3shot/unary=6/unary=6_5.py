
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = torch.nn.ReLU()
    def forward(self, x1, x2, x3, x4, x5, x6):
        h1 = self.relu(x1)
        h2 = self.relu(x2)
        h3 = self.relu(x3)
        h4 = self.relu(x4)
        h5 = self.relu(x5)
        h6 = self.relu(x6)
        return (h1+h2) / 2 + (h3+h4) / 2 + (h5+h6) / 2
# Inputs to the model
x1 = torch.randn(1, 1, 120, 120)
x2 = torch.randn(1, 1, 120, 120)
x3 = torch.randn(1, 1, 120, 120)
x4 = torch.randn(1, 1, 120, 120)
x5 = torch.randn(1, 1, 120, 120)
x6 = torch.randn(1, 1, 120, 120)
