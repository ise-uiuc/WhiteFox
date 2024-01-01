
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = torch.randn(4, 1)
        self.b = torch.randn(4, 1)
        self.c = torch.randn(4, 1)
    def forward(self, x):
        x = torch.mm(self.a, x)
        x = torch.cat((x, x), dim=1)
        x = torch.mm(self.b, x)
        x = torch.mm(self.c, x)
        return x
# Inputs to the model
x = torch.randn(2, 1)
