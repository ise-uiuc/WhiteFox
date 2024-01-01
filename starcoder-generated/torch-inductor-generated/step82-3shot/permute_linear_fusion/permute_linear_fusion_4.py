
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
        self.concat = torch.cat()
        self.add = torch.add()
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v2 = torch.nn.functional.linear(x=v1, weight=self.linear.weight, bias=self.linear.bias)
        v3 = self.concat(v2, self.add(v1, v1))
        return v3
# Inputs to the model
x1 = torch.randn(1, 2, 2)
