
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(2, 1)
    def forward(self, x1):
        v1 = torch.nn.functional.linear(x1, self.linear1.weight, self.linear1.bias)
        return v1.permute(0, 2, 1)
# Inputs to the model
x1 = torch.randn(2, 2, 2)
