
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        v1 = torch.bmm(x1.permute(0, 2, 1), x2).permute(0, 2, 1)
        v2 = torch.bmm(x2.permute(0, 2, 1), x1).permute(0, 2, 1)
        return (v1, v1, x1, x2, v2)
# Inputs to the model
x1 = torch.ones(1, 4, 4)
x2 = torch.ones(1, 4, 4)
