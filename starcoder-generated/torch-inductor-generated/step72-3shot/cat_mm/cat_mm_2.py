
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.ReLU()
    def forward(self, x1, x2):
        v1 = x1.t().contiguous()
        v2 = x2.t().contiguous()
        v1 = self.linear(v1)
        v2 = self.linear(v2)
        return torch.cat((v1, v2), 0).t()
# Inputs to the model
x1 = torch.randn(2, 5, 4)
x2 = torch.randn(4, 5, 6)
