
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 1)
    def forward(self, x1):
        v1 = torch.nn.functional.linear(x1, self.linear.weight, self.linear.bias)
        v2 = v1.permute(0, 1, 3, 2)
        v3 = v2.contiguous()
        v4 = v3.view(7, 3)
        return v4.view(1, 3, 7)
# Inputs to the model
x1 = torch.randn(1, 3, 7, device='cpu')
