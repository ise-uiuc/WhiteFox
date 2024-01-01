
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(1, 3, bias=False)
        self.linear2 = torch.nn.Linear(3, 1)
    def forward(self, x1):
        v1 = torch.nn.functional.linear(x1, self.linear1.weight)
        v2 = v1.permute(0, 2, 1)
        v3 = torch.nn.functional.linear(v1, self.linear2.weight)
        return v3
# Inputs to the model
x1 = torch.randn(2, 3, 3)
