
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2, bias=False)
    def forward(self, x1):
        v3 = torch.nn.functional.linear(x1, self.linear.weight).permute(0, 1, 3, 2)
        return torch.nn.functional.linear(v3, self.linear.weight)
# Inputs to the model
x1 = torch.randn(1, 2, 2, 2, device='cpu')
