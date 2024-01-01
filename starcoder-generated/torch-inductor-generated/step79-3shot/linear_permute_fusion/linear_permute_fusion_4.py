
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, x1):
        v1 = x1 + 1.0
        v2 = v1.permute(0, 2, 1)
        v4 = v2.size()
        v7 = v4[1]
        v8 = v4[-1]
        v9 = v4[-2]
        v5 = v7 + 10
        v10 = v8 + 10
        v6 = v9 + 10
        v3 = torch.nn.functional.linear(v5, self.linear.weight[0:v10,:,:], self.linear.bias[0:v6,:])
        return v3
# Inputs to the model
x1 = torch.randn(2, 4, 2)
