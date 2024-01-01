
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        return torch.cat([torch.mm(x, x) for x in [
            torch.cat([x1, x1, x1], 1),
            torch.cat([x1, x1, x1], 1),
            torch.cat([x1, x1, x1], 1),
            torch.cat([x1, x1, x1], 1),
            torch.cat([x1, x1, x1], 1),
            ]], 1)
# Inputs to the model
x1 = torch.randn(3, 3)
