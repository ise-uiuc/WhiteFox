
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.query = torch.nn.Linear(256, 256)
        self.key = torch.nn.Linear(32, 256)
        self.value = torch.nn.Linear(256, 64)
        self.dropout = torch.nn.Dropout(p=0.2)
 
    def forward(self, q, k, v):
        q = self.query(q)
        k = self.key(k)
        v = self.value(v)
        qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = qk.mul(1.0 / math.sqrt(q.size(-1)))
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = self.dropout(softmax_qk)
        output = torch.matmul(dropout_qk, v)
        return output

# Initializing the model
m = Model()

# Inputs to the model
q = torch.randn(1, 256, 1)
k = torch.randn(1, 32, 1)
v = torch.randn(1, 256, 1)
