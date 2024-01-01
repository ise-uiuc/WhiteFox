
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear0 = torch.nn.Linear(2, 2)
        self.linear1 = torch.nn.Linear(2, 2)
        self.elu = torch.nn.ELU(alpha=0, inplace=True)
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v2 = torch.nn.functional.linear(v1, self.linear0.weight, self.linear0.bias)
        v3 = torch.nn.functional.linear(v2, self.linear1.weight)
        t0 = v2 + v3
        t1 = torch.sigmoid(t0)
        t2 = t1 * v2
        t3 = t0 - t2
        t4 = self.elu.forward(t2)
        t5 = self.elu.forward(t3)
        v4 = t4 + t5
        return v4
# Inputs to the model
x1 = torch.randn(1, 2, 2)
