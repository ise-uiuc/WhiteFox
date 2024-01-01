
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.pointwise = torch.nn.Conv2d(1, 1, 1, stride=1, padding=0)
        self.linear = torch.nn.Linear(1, 1)
    def forward(self, x1):
        v1 = self.pointwise(x1)
        v2 = torch.sigmoid(v1)
        v3 = self.linear(v2)
        v4 = torch.tanh(v1)
        v5 = self.pointwise(x1)
        v6 = torch.softmax(v2)
        return v2
# Inputs to the model
x1 = torch.randn(1, 1, 64, 64)
