
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = torch.nn.Dropout(.1)

    def forward(self, q, k, v, a):
        w = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))
        w = w + a
        w = torch.softmax(w, dim=-1)
        w = self.dropout(w)
        w = w @ v
        w = torch.tanh(w)
        return w

# Initializing the model
m = Model()

# Inputs to the model
q = torch.randn(1, 32, 300)
k = torch.randn(1, 32, 300)
v = torch.randn(1, 32, 128)
a = torch.randn(1, 1, 1, 300)
