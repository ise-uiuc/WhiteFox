
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v2 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
        v3 = v2.detach()
        v4 = torch.linalg.norm(v2, ord=0)
        z1 = v3 * v4
        z1 = z1.detach()
        return z1
# Inputs to the model
x1 = torch.randn(1, 2, 2)
