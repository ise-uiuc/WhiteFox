
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 3)
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v1 = torch.pow(v1, 2)
        v2 = torch.nn.functional.linear(v1, self.linear.weight, None)
        v3 = torch.sqrt(2)
        return v3 * v2
# Inputs to the model
x1 = torch.randn(1, 2, 2)
