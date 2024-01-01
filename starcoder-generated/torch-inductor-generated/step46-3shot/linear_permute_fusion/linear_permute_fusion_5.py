
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 3)
        self.act = torch.nn.modules.activation.Sigmoid()
    def forward(self, x1):
        v1 = torch.nn.functional.linear(x1, self.linear.weight, self.linear.bias)
        v2 = self.act(v1)
        v3 = v2.permute(1, 0)
        v5 = v3.clone().detach()
        v4 = v3.permute(1, 0)
        v7 = v5.add(v4)
        return v7 + v2
# Inputs to the model
x1 = torch.randn(3, 2)
