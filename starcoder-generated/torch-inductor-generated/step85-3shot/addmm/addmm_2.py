
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, bias, A, B):
        x = torch.mm(x, A)
        bias = torch.mm(bias, B)
        return x + bias
# Inputs to the model
x = torch.randn(10, 10, requires_grad=True)
bias = torch.randn(10, 10, requires_grad=True)
A = torch.randn(10, 10, requires_grad=True)
B = torch.randn(10, 10, requires_grad=True)
