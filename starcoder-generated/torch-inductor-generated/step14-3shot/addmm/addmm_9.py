
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
    def forward(self, x, w, b):
        v1 = torch.mm(w, x)
        v2 = v1 + b
        return v2
# Inputs to the model
x = torch.randn(12, 6)
w = torch.randn(12, 6)
b = torch.randn(2, 4)
