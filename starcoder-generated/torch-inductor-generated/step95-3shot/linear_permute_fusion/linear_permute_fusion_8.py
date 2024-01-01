
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(2, 2)
        self.linear2 = torch.nn.Linear(2, 2)
    def forward(self, x1):
        v1 = torch.nn.functional.linear(x1, self.linear2.weight, self.linear2.bias).permute(0, 2, 1)
        v2 = v1.permute(0, 2, 1)
        return v2.permute(0, 2, 1)
# Inputs to the model
x1 = torch.randn(1, 2, 2)
