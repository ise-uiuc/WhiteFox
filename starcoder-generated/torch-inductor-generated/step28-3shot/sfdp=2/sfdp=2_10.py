
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.query_key = Linear(1024, 1024)
        self.softmax_dropout_value = Linear(1024, 1024)

    def forward(self, x1, x2):
        k = self.query_key(x1)
        v = self.softmax_dropout_value(x2)
        d = torch.matmul(q, k.transpose(-2, -1))
        inv_d = 1.0 / math.sqrt(d.size()[-1])
        d = torch.div(d, inv_d)
        s = d.softmax(dim=-1)
        dropout_s = torch.nn.functional.dropout(s, p=0.3)
        o = dropout_s.matmul(v)
        return o

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 1024)
x2 = torch.randn(1, 1024)
