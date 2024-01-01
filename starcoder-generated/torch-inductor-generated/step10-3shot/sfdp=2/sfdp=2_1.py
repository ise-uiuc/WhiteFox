
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.q = torch.nn.Linear(1024, 1024)
        self.k = torch.nn.Linear(1024, 1024)
        self.v = torch.nn.Linear(1024, 1024)
        self.dropout = torch.nn.Dropout(0.25)
        self.scale_factor = 10.0
 
    def forward(self, query, key, value):
        q = self.q(query)
        k = self.k(key)
        v = self.v(value)
        inv_scale_factor = 1.0 / self.scale_factor
        qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = self.dropout(softmax_qk)
        output = dropout_qk.matmul(v)
        return output, softmax_qk

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(1, 256, 1024)
attn_mask = torch.randn(1, 256, 256) * math.exp(-1e4)
key = key = query + attn_mask
value = torch.randn(1, 256, 1024)
