
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 10)
    def forward(self, x1):
        v1 = self.linear(x1)
        return nn.Sigmoid()(v1)
# Inputs to the model
x1 = torch.randn(1, 3)
