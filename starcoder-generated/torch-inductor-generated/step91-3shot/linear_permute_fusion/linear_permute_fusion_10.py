
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_1 = torch.nn.Linear(2, 8)
        self.linear_2 = torch.nn.Linear(2, 2)
        self.linear_3 = torch.nn.Linear(2, 3)
    def forward(self, x5):
        v0 = x5
        v1 = torch.nn.functional.linear(v0, self.linear_1.weight, self.linear_1.bias)
        v2 = v1.permute(0, 2, 1)
        v3 = v2.permute(0, 1, 3)
        v4 = v3.contiguous()
        v5 = torch.nn.functional.linear(v3, self.linear_2.weight, self.linear_2.bias)
        v6 = v5.permute(0, 2, 1)
        v7 = v6.contiguous()
        v8 = v7.view(1, 3, 6)
        v9 = torch.nn.functional.linear(v8, self.linear_3.weight, self.linear_3.bias)
        return v9
# Inputs to the model
x5 = torch.randn(1, 2, 2)
