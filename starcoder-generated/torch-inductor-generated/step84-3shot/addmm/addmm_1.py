
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.inp = torch.randn(3, 3)
    def forward(self, x1, x2):
        inp = x1
        input1 = torch.mm(inp, self.inp) # Pass the input tensor to another method so that it can be used for another operation
        return input1 + x2
# Inputs to the model
x1 = torch.randn(3, 3, requires_grad=True)
x2 = torch.randn(3, 3)
