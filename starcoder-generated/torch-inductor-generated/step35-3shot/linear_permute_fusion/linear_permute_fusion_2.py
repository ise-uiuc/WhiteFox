
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, x1):
        y = torch.randperm(1)
        v1 = torch.nn.functional.linear(x1, self.linear.weight, self.linear.bias)
        v2 = v1.chunk(1, dim=(y-1))
        return v2[0].squeeze(dim=-2)
# Inputs to the model
x1 = torch.rand(2, 2, 2, 2, 2, 2)
