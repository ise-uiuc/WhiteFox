
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        for i in range(21):
            a = torch.mm(x1, x2)
            b = torch.mm(x1, x2)
        return torch.cat(a, b, 1)
# Inputs to the model
x1 = torch.randn(3, 4)
x2 = torch.randn(4, 3)
