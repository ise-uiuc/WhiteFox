
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v2 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
        v3 = torch.mean(v2, dim=[0, 1])
        v3 = v3.view(1, 1, 1)
        return torch.repeat_interleave(v3, 2, 0)
# Inputs to the model
x1 = torch.randn(3, 2, 2)
