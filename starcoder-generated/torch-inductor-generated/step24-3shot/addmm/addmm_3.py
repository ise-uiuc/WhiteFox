
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        v1 = torch.mm(x2, x1) # The second input tensor is passed as the first input in the operation
        return v1
# Inputs to the model
x1 = torch.randn(1, 1)
x2 = torch.randn(1321, 2)
