
class Model(torch.nn.Module):
    def __init__(self, query_size, key_size):
        super().__init__()
        self.query = torch.nn.Linear(query_size, 1)
        self.key = torch.nn.Linear(key_size, 1)
 
    def forward(self, query, key, dropout_p):
        q = self.query(query)
        k = self.key(key)
        scale_factor = np.sqrt(2.0 / (q.shape[-1] * k.shape[-1]))
        qk = torch.mul(q, k.transpose(-2, -1))
        scaled_qk = qk * scale_factor
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = F.dropout(softmax_qk, p=dropout_p)
        v = dropout_qk.matmul(v)
        v = self.head_output(v)
        return v
 
# Initializing the model
m = Model(query_size, key_size)

# Inputs to the model
query = torch.randn(1, 8)
key = torch.randn(1, 48)
dropout_p = 0.1
v = torch.randn(1, 48)
