
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        return torch.cat( torch.mm(x1, x1).tolist() * 5, 1)
# Inputs to the model
x1 = torch.randn(1, 2)
