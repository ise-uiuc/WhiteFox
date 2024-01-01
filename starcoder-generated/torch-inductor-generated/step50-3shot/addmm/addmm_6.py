
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, inp):
        # Replace the method with any method you like that accepts the same input tensors and arguments as torch.mm
        v1 = torch.mm(x1, x2) + inp
        return v1
# Inputs to the model
x1 = torch.randn(3, 3)
x2 = torch.randn(3, 3)
inp = torch.randn(3, 3)
