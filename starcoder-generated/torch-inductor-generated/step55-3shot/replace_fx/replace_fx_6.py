
class model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x1 = torch.rand_like(input=x, size=(1, 3, x.size(1), x.size(2)))
        x2 = torch.rand_like(input=x, size=(1, x.size(0), x.size(1) * x.size(2)))
        x3 = torch.rand_like(input=x, size=x.size())
        x4 = torch.rand_like(input=x, size=(1,3,x.size(1)*x.size(2)))
        return x1 + x2 + x3 + x4
# Inputs to the model
x = torch.rand((1, 2, 3, 4))
