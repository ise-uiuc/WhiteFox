
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, x1):
        v14 = []
        v13 = []
        v4 = x1
        v26 = torch.nn.functional.linear(v4, self.linear.weight, self.linear.bias)
        v12 = v26.permute(0, 2, 1)
        v25 = v12.contiguous()
        v14.append(v25)
        v13.append(v14)
        v24 = v13[0][0]
        return v24
# Inputs to the model
x1 = torch.randn(1, 2, 2)
