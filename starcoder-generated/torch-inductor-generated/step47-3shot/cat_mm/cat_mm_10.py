
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        x1_f = x1.flatten(0, 1)
        x2_f = x2.flatten(0, 1)
        x3_f = torch.cat([x1, x1, x1], dim=0)
        # Do another torch.cat along dimension 0
        return torch.cat([x1], dim=0)
# Inputs to the model
x1 = torch.randn(2, 3, 2)
x2 = torch.randn(2, 2, 5)
