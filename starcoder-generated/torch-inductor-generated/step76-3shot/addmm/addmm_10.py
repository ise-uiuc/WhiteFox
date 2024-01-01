
inp = torch.randn(3, 3)
x1 = torch.randn(3, 3, requires_grad=True)
def forward(self, x2):
    v1 = torch.mm(inp, x1)
    return v1 + x2
# Inputs to the model
x2 = torch.randn(3, 3)
