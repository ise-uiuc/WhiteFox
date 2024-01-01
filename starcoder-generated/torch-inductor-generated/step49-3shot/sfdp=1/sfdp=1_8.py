
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x1):
        k = torch.randn(1, 9, 24)
        t2 = torch.matmul(x1, k.transpose(0, 1))
        v = torch.randn(24, 24)
        scale = 1.0 / math.sqrt(1)
        t1 = t2.div(scale)
        t3 = torch.softmax(t1)
        t4 = torch.nn.functional.dropout(t3)
        out = torch.matmul(t4, value)

        return out

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(24, 9)
