
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 4)
    def forward(self, x2):
        v2 = x2
        v4 = False
        if v4:
            v3 = torch.nn.functional.linear(v2, self.linear.weight, self.linear.bias)
            v6 = x2 - v3
            v7 = v3.permute(0, 2, 1)
            v8 = v7.flip(0)
            return v6 + v8
        else:
            v3 = torch.nn.functional.linear(v2, self.linear.weight, self.linear.bias)
            v6 = x2 - v3
            v7 = v3
            v8 = v7.permute(0, 2, 1)
            return v6 + v8.flip(0)
# Inputs to the model
x2 = torch.randn(1, 4, 4)
