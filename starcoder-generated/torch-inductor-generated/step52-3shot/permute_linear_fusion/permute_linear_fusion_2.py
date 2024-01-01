
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, x1, x2):
        v3 = torch.stack((x1, x2), dim=1)
        v3 = torch.flip(v3, dims=(1,))
        v4 = v3.sum(dim=-2)
        v3 = v3.permute(0, 2, 1)
        v3 = torch.nn.functional.linear(v3, self.linear.weight, self.linear.bias)
        v5 = v3.permute(0, 2, 1)
        v5 = torch.nn.functional.linear(v5, self.linear.weight, self.linear.bias)
        return (v3, v5), v4
# Inputs to the model
x1 = torch.randn(1, 2, 2)
x2 = torch.randn(1, 2, 2)
