
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, y):
        v1 = torch.mm(x, y)
        v2 = v1 + torch.mm(x, y)
        return v1
# Inputs to the model
x = torch.randn(3, 3, requires_grad=True)
y = torch.randn(3, 3)
