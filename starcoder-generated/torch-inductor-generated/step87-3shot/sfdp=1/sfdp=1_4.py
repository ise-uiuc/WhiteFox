
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a, b, c, d, e, f):
        w1 = torch.matmul(a, b.transpose(-2, -1))
        w2 = w1.div(d)
        w3 = w2.softmax(dim=-1)
        w4 = torch.nn.functional.dropout(w3, f)
        v1 = torch.matmul(w4, c)
        return v1

# Initializing the model
m = Model()

# Inputs to the model
w1 = torch.randn(1, 64, 100)
w2 = torch.randn(1, 100, 128)
w3 = torch.randn(1, 128, 256)
w4 = torch.randn(1, 256, 20)
