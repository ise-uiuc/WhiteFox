
class Model(torch.nn.Module):
    def __init__(self, hidden_size, num_heads, dropout_p):
        super().__init__()
        self.qkv_proj = torch.nn.Linear(hidden_size, hidden_size *2, bias=False)
        self.dropout_p = dropout_p
 
    def forward(self, query, key, value, inv_scale_factor):
        qkv = self.qkv_proj(query).reshape(query.shape[0], -1, 2, np.power(self.n_heads, -1).astype(int))
        q, k, v = torch.chunk(qkv, chunks=3, dim=-1)
        q = q.reshape(q.shape[0], q.shape[1], q.shape[2])
        k = k.reshape(k.shape[0], k.shape[1], k.shape[2])
        v = v.reshape(v.shape[0], v.shape[1], v.shape[2])
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
hidden_size = 128
num_heads = 4
dropout_p = 0.2
m = Model(hidden_size, num_heads, dropout_p)

# Inputs to the model
query = torch.randn(32, 128)
key = torch.randn(32, 256)
value = torch.randn(32, 256)
inv_scale_factor = torch.ones((32,))
