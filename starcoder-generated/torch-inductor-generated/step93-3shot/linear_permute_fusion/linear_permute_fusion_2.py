
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 1)
    def forward(self, x1):
        v6 = True
        v1 = torch.nn.functional.linear(x1, self.linear.weight, self.linear.bias)
        v2 = v1.permute(0, 2, 1)
        v4 = x1 - v2
        v5 = True
        #if v5:
        #    v3 = torch.nn.functional.linear(x1, self.linear.weight, self.linear.bias)
        if v6:
            v3 = v1 + v4
            return v3
        else:
            v3 = v1 + v4
            return v3
# Inputs to the model
x1 = torch.randn(1, 2, 2)
