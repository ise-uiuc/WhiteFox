
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, input2, input1):
        inp1 = torch.mm(x1, input2)
        inp2 = torch.mm(inp1, input1) + input2
        return inp2
# Inputs to the model
x1 = torch.randn(1, 1321)
input2 = torch.randn(1321, 1321)
input1 = torch.randn(1321, 1)
