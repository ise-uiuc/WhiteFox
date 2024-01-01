
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        output = torch.mm(x1, x2)
        return output
# Inputs to the model
x1 = torch.randn(7, 5)
x2 = torch.randn(5, 3)
