
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, x1):
        v1 = torch.nn.functional.linear(x1, self.linear.weight, self.linear.bias)
        v2 = v1.permute(0, 1, 3, 2)
        for i in range(3):
            if i == 0:
                v3 = torch.nn.functional.linear(x1, self.linear.weight, self.linear.bias)
                v4 = v3.permute(0, 2, 1, 3)
                continue
            v1 = torch.nn.functional.linear(x1, self.linear.weight, self.linear.bias)
            v2 = v1.permute(0, 1, 3, 2)
        return v4
# Inputs to the model
x1 = torch.randn(1, 2, 2, 2, device='cpu')
