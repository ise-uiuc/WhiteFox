
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
        self.softmax = torch.nn.Softmax(dim=1)
    def forward(self, x1):
        v5 = x1
        v1 = torch.nn.functional.linear(v5, self.linear.weight, self.linear.bias)
        v2 = v1.permute(0, 1, 3, 2)
        v3 = self.softmax(v2)
        v4 = v3.to(dtype=torch.double)
        return v4
# Inputs to the model
x1 = torch.randn(1, 2, 2, 2, device='cpu', requires_grad=True)
