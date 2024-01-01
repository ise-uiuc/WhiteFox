
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(2, 2)
    def forward(self, x1):
        v1 = torch.nn.functional.linear(x1, self.linear1.weight, self.linear1.bias)
        v2 = torch.div(v1, 3.0)
        v3 = v2.permute(0, 2, 1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 2, 2)
