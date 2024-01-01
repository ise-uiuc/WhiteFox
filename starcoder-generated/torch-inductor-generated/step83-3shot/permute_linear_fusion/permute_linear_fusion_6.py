
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear0 = torch.nn.Linear(2, 2)
        self.linear1 = torch.nn.Linear(2, 2)
        self.elu = torch.nn.ELU(alpha=0, inplace=True)
        self.tanhshrink = torch.nn.Tanhshrink()
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v2 = torch.nn.functional.linear(v1, self.linear0.weight, self.linear0.bias)
        v3 = torch.tanh(torch.nn.functional.linear(v2, self.linear1.weight))
        v4 = self.elu(v2) * self.tanhshrink(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 2, 2)
