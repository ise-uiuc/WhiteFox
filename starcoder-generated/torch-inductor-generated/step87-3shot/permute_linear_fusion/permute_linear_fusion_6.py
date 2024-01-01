
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2, bias=False)
        self.relu = torch.nn.ReLU6()
    def forward(self, x1):
        v1 = x1.permute(0, 3, 1, 2)
        v2 = self.relu(v1)
        v3 = torch.nn.functional.linear(v1, self.linear.weight)
        v4 = v2 + v3
        return v4
# Inputs to the model
x1 = torch.randn(1, 2, 3, 3)
