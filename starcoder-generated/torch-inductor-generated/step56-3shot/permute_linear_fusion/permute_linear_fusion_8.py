
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 3)
    def forward(self, x1):
        x1 = self.linear(x1)
        x3 = torch.max(x1, dim=-1)[1]
        x3 = torch.nn.functional.softmax(x1 + 1.329)
        x4 = x3.dim()
        x1 = self.linear(x1)
        x4 = torch.max(x1, dim=-1)[0]
        return x4
# Inputs to the model
x1 = torch.randn(1, 3)
