
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 1)
    def forward(self, x1):
        v4 = x1
        v5 = False
        if v5:
            v2 = False
            if v2:
                v1 = torch.nn.functional.linear(v4, self.linear.weight, self.linear.bias)
                v3 = v1.permute(0, 2, 1) + v1.permute(0, 1, 2)
            else:
                v1 = torch.nn.functional.linear(v4, self.linear.weight, self.linear.bias)
                v3 = v1.permute(0, 2, 1) - v1.permute(0, 1, 2)
        else:
            v1 = torch.nn.functional.linear(v4, self.linear.weight, self.linear.bias)
            v3 = v1.permute(0, 2, 1) + 2*v1.permute(0, 1, 2) + v1.permute(0, 2, 1) + 3*v1.permute(0, 1, 2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 2, 2)
