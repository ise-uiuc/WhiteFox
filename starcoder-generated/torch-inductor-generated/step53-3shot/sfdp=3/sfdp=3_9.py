
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, __x1__, __x2__, __x3__):
        v1 = torch.matmul(__x1__, __x2__.transpose(-2, -1))
        v2 = v1 * 10000.000000
        v3 = v2.softmax(dim=-1)
        v4 = torch.nn.functional.dropout(v3, p=0.30000001192092896)
        v5 = v4.matmul(__x3__)
        return v5

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 10, 10)
x2 = torch.randn(1, 10, 10)
x3 = torch.randn(1, 10, 10)
