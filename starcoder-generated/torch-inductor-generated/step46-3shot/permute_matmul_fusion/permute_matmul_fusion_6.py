
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.A = torch.nn.Parameter(torch.ones(4, 5, 6))
        self.B = torch.nn.Parameter(torch.ones(4, 6, 7))

    def forward(self, x):
        out = torch.bmm(self.A, self.B)
        return out + x
# Inputs to the model
x = torch.randn(4, 5, 7)
