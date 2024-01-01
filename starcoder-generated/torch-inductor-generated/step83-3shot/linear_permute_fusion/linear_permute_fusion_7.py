
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 4)
    def forward(self, x0):
        v0 = x0
        v1 = torch.nn.functional.linear(v0, self.linear.weight, torch.tensor(10))
        v2 = v1.permute(0, 2, 1)
        return v2
# Inputs to the model
x0 = torch.randn(1, 2, 2)
