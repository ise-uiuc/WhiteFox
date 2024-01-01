
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 3, bias=True)
    def forward(self, x1):
        v3 = x1.permute(0, 2, 1)
        v0 = torch.nn.functional.linear(v3, self.linear.weight)
        v1 = self.linear.weight
        v2 = v0 * v1
        return v2
# Inputs to the model
x1 = torch.randn(1, 2, 2, device='cpu')
