
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
        self.ReLU = torch.nn.ReLU()
    def forward(self, x1):
        v1 = x1 + torch.Tensor([2.5,.3, -.7,.1, -2.3])
        v2 = v1.permute(0, 2, 1)
        v3 = torch.nn.functional.linear(v2, self.linear.weight, self.linear.bias)
        x2 = self.ReLU(v3)
        v4 = x2.detach()
        v5 = torch.max(v4, dim=-1)[1]
        return (self.linear.bias)[v5]
# Inputs to the model
x1 = torch.randn(1, 2, 2)
