
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 2, device='cpu')
    def forward(self, x1):
        v3 = self.linear.weight.permute(0, 2, 1)
        v1 = torch.nn.functional.linear(x1, self.linear.weight, self.linear.bias)
        v5 = x1.transpose(1, 2)
        v2 = v1.permute(0, 2, 1)
        return v2
# Inputs to the model
x1 = torch.randn(3, 2)
