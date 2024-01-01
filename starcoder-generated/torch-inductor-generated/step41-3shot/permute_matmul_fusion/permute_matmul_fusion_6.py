
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        x1 = x1 + 2
        x2 = x2 - 3
        o1 = x1.permute(2, 0, 1)
        o2 = x2.permute(2, 1, 0)
        o3 = o2 @ o2
        return o3.permute(2, 2, 0)
# Inputs to the model
x1 = torch.zeros((2, 4, 3))
x2 = torch.zeros((5, 3, 2))
