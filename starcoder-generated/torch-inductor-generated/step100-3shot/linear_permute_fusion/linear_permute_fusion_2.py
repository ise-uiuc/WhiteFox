
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, x1):
        v1 = torch.nn.functional.linear(x1, self.linear.weight, self.linear.bias)
        v2 = v1.squeeze(0).permute(1, 0)
        return torch.nn.functional.linear(v2, torch.ones(2,2))
# Inputs to the model
x1 = torch.randn(1, 1, 2, 2)
