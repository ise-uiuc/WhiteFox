
class model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        t1 = torch.randn_like(x1)
        return t1
# Inputs to the model
x1 = torch.randn((8, 8))
