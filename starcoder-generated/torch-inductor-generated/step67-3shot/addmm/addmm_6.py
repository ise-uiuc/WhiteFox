
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        x1 = x1 * x1 # 'x1' is the same object as the input 'x1'
        x2 = x1 + x1 # 'x2' is the same object as the input 'x1'
        x3 = x2 + x2 # 'x3' is the same object as the input and the output 'x2'
        x2 = x3 + x3 # 'x2' is the same object as the output 'x3'
        return x3
# Inputs to the model
x1 = torch.randn(3, 3, requires_grad=True)
