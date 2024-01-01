
class Model1(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, y):
        v1 = torch.mm(x1, y)
        return v1
# Inputs to the model
x1= torch.randn(5, 3)
x2 = torch.randn(3, 3, requires_grad=True)
y = torch.randn(3, 3)
