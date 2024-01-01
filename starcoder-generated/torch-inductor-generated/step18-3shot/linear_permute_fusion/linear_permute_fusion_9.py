
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, x1):
        v4 = x1.to('cpu')
        v1 = 5
        v2 = v1 + 6
        v4 = v4.to('cpu')
        v2 = torch.nn.functional.linear(v4, self.linear.weight, self.linear.bias)
        v5 = v2.permute(0, 2, 1)
        v4 = v4.to('cpu')
        v5 = v2.permute(0, 2, 1)
        v6 = v5.to('cpu')
        v7 = torch.nn.functional.linear(v6, self.linear.weight, self.linear.bias)
        return v6
# Inputs to the model
x1 = torch.randn(1, 2, 2, device='cuda')
