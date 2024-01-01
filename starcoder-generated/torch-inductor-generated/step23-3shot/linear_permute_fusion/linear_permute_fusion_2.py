
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 4)
    def forward(self, x1):
        v1 = self.linear(x1)
        t1 = v1[..., :2].t()
        t2 = v1[..., 2:].permute(0, 3, 2, 1)
        return t1 + t2
# Inputs to the model
x1 = torch.randn(1, 2, 2, 2)
