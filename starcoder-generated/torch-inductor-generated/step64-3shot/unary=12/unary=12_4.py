
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = torch.nn.ReLU()
        self.pool = torch.nn.AvgPool2d(3, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.pool(self.relu(x1))
        v2 = v1 - 0.5
        v3 = torch.relu(v2)
        v4 = v3 / 0.01 
        return v4
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
