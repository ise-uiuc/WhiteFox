
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, input1, input2):
        v1 = torch.mm(x1, input2)
        return v1 + x2 # Replaced inp by x2
# Inputs to the model
x1 = torch.randn(3, 3)
x2 = torch.randn(3, 3)
input1 = torch.randn(3, 3, requires_grad=True)
input2 = torch.randn(3, 3, requires_grad=True)
