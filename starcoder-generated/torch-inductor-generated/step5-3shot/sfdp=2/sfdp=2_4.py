
class Model(torch.nn.Module):
    def __init__(self, n_head, d_qkv, dropout_p):
        super().__init__()
        self.query = torch.nn.Linear(d_qkv, d_qkv)
        self.key = torch.nn.Linear(d_qkv, d_qkv)
        self.value = torch.nn.Linear(d_qkv, d_qkv)
        self.dropout = torch.nn.Dropout(dropout_p)
 
    def forward(self, query, key, value):
        q = self.query(query)
        k = self.key(key)
        v = self.value(value)
 
        qk = torch.matmul(q, k.transpose(-2, -1))
        inv_scale_factor = (1.0 / math.sqrt(d_qkv))  # 1d tensor of size d_qkv
        scaled_qk = qk * inv_scale_factor
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = self.dropout(softmax_qk)
        output = dropout_qk.matmul(v)
        return output

# Initializing the model
m = Model(n_head, d_qkv, dropout_p)

# Inputs to the model
query = torch.randn(1, n_head, n_query_seq, d_qkv)
key = torch.randn(1, n_head, n_kv_seq, d_qkv)
value = torch.randn(1, n_head, n_kv_seq, d_qkv)
