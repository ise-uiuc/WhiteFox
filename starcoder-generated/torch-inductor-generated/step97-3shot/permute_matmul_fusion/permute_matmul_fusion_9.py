
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x2):
        v5 = x2.permute(1, 2, 0)
        return torch.bmm(x2, v5)
# Inputs to the model
x2 = torch.randn(1, 2, 2)
