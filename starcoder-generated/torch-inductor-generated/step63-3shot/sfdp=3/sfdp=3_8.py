
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.key = torch.nn.Linear(8, 8)
        self.query = torch.nn.Linear(8, 8)
        self.value = torch.nn.Linear(8, 8)
 
    def forward(self, k, q):
        qk = (self.query(q).mul(-2).exp() @ self.key(k).transpose(-1, -2))
        scale_factor = 1 / (1 + torch.arange(qk.size(-1)))
        softmax_qk = qk * scale_factor
        dropout = torch.nn.functional.dropout(softmax_qk, p=0.2)
        output = dropout @ self.value(v)
        return qk, output

# Initializing the model
m = Model()

# Inputs to the model
k = torch.randn(1, 8)
q = torch.randn(1, 8)
v = torch.randn(1, 8)

qk, output = m(k, q, v)

