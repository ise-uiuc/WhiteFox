
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Sequential()
        self.linear.add_module("flatten", torch.nn.Flatten(0, 1))
        self.linear.add_module("linear", torch.nn.Linear(2, 2))
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1.squeeze(-1)
        v3 = v2.transpose(1, 2)
        v4 = torch.sum(v3, dim=1, keepdim=True)
        return v4
# Inputs to the model
x1 = torch.randn(1, 2, 2)
