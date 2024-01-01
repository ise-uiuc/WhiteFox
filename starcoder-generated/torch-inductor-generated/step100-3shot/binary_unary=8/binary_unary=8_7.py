
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = torch.nn.ReLU()
    def forward(self, x1, x2):
        v1 = self.relu(x2)
        v2 = self.relu(x2)
        v3 = self.relu(x2)
        v4 = self.relu(x2)
        v5 = v1 + v2 + v3 + v4
        return v5
# Inputs to the model
x1 = torch.randn(1, 1280)
x2 = torch.randn(1, 256)
