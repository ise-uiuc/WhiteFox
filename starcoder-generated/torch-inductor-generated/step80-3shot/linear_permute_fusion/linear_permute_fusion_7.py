
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 2)
    def forward(self, x):
        # Linear transformation followed by a bias add
        v1 = torch.nn.functional.linear(x, self.linear.weight, self.linear.bias)
        # Element-wise multiplication
        v2 = torch.mul(x, v1)
        # Sum over the last dimension
        v3 = torch.sum(v2, dim=2, keepdim=False)
        return v3
# Inputs to the model
x = torch.randn(1, 2, 2, 2)
