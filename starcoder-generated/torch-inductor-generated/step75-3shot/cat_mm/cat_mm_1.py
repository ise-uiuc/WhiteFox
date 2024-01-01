
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        v = torch.cat([torch.cat([x1, x2]) for i in range(4)], 2)
        return v
# Inputs to the model
x1 = torch.randn(2, 2, 4)
x2 = torch.randn(2, 2, 2)
