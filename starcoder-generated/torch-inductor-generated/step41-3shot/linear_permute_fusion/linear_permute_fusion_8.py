
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, x1):
        v1 = torch.nn.functional.linear(x1, self.linear.weight, self.linear.bias)
        v2 = v1.permute(0, 2, 1)
        return torch.Tensor.permute(v1, 0, 1)
# Inputs to the model
x1 = torch.randn(5, 2, 2, device='cpu', dtype=torch.float16)
