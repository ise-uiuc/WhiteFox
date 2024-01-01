
class Model(torch.nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        self.W_qkv = torch.nn.Linear(dim, 3 * dim)
        self.n_heads = n_heads
 
    def forward(self, query, key, value):
        qkv = self.W_qkv(query).reshape(query.shape[0], query.shape[1], 3, self.n_heads, self.dim)
        qkv = qkv.transpose(1, 2)
        q, k, v = qkv[0], qkv[1], qkv[2]
        scale_factor = torch.sqrt(torch.tensor(1/(self.dim*self.n_heads)))
        qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = qk.div(scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=0.0, training=True)
        output = dropout_qk.matmul(v).transpose(1, 2)
        output = output.reshape(output.shape[0], output.shape[1], -1)
        return output

# Initializing the model
m = Model(dim=64, n_heads=8)

# Inputs to the model
query = torch.randn(1, 8, 64)
key = torch.randn(1, 4, 64)
value = torch.randn(1, 4, 64)
output = m(query, key, value)

