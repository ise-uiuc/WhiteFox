
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 2)
    def forward(self, x1):
        v1 = torch.nn.functional.linear(x1, self.linear.weight, self.linear.bias)
        v2 = v1.reshape(1, 2, 3)
        v3 = v2.permute(0, 2, 1)
        v4 = v3.reshape(2, 6)
        v5 = v2.reshape(2, 3)
        v6 = torch.cat((v4, v5), 1)
        v7 = v6.transpose(0, 1)
        return v7.reshape(3, 2)
# Inputs to the model
x1 = torch.randn(1, 3, 3)
