
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        v1 = torch.randn(4, 4, requires_grad=True)
        v2 = torch.randn(3, 4, requires_grad=True)
        v3 = torch.randn(4, 3, requires_grad=True)
        v4 = torch.randn(3, 4, requires_grad=True)
        v5 = torch.randn(4, 4, requires_grad=True)
        v6 = torch.randn(3, 3, requires_grad=True)

        v7 = torch.mm(v1, v2)
        v8 = torch.mm(v3, v4)
        v9 = torch.mm(v5, v6)

        output = torch.cat([v7, v8], 1)
        output = torch.cat([output, v9], 1)
        return output

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 8, 8)
