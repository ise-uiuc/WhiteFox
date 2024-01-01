
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_1 = torch.nn.Linear(2, 2)
        self.linear_2 = torch.nn.Linear(2, 4)
        self.linear_3 = torch.nn.Linear(2, 1)
    def forward(self, x6):
        v0 = x6
        v1 = torch.nn.functional.linear(v0, self.linear_1.weight, self.linear_1.bias)
        v2 = v1.permute(0, 2, 1)
        v3 = v2.contiguous()
        v4 = torch.nn.functional.linear(v3, self.linear_2.weight, self.linear_2.bias)
        v5 = v4.reshape(1, 4, 1).mean(2)
        v6 = torch.nn.functional.linear(v5, self.linear_3.weight, self.linear_3.bias)
        v7 = v6.reshape(1, 1, 1)
        return v7
# Inputs to the model
x6 = torch.randn(1, 2, 2)
