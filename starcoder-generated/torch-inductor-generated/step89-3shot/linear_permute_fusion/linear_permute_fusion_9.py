
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 3)
    def forward(self, x1):
        v1 = torch.nn.functional.linear(x1, self.linear.weight, self.linear.bias)
        v2 = v1 + v1
        v3 = v2 + v2
        v5 = torch.nn.functional.softmax(v3, dim=0)
        transpose1 = torch.transpose
        v6 = transpose1(v5, 0, 1)
        softmax1 = torch.nn.functional.softmax
        v7 = softmax1(v3, dim=-1)
        v8 = v7.permute(2, 1, 0)
        v9 = torch.flatten(v8)
        v10 = v9.unsqueeze(1)
        v11 = v10.reshape(-1)
        v12 = v11 + v11
        return v12
# Inputs to the model
x1 = torch.randn(1, 2, 2)
