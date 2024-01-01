
# Note: The two inputs do not have the same number of rows.
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, inp1, inp2):
        # Perform matrix multiplication by transposing the second input
        v1 = torch.mm(x1, inp1)
        v2 = torch.mm(v1, x2.T)
        return v2
# Inputs to the model
x1 = torch.randn(2, 4)
x2 = torch.randn(0, 4)
inp1 = torch.randn(2, 4)
inp2 = torch.randn(2, 2)
