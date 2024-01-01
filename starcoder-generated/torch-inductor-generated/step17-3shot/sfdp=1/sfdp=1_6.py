
class Model(torch.nn.Module):
    def __init__(self, dim, num_heads, dropout_rate):
        super().__init__()
        self.query = torch.nn.Linear(dim, dim)
        self.key = torch.nn.Linear(dim, dim)
        self.value = torch.nn.Linear(dim, dim)
        self.dropout = torch.nn.Dropout(p=dropout_rate)
 
    def forward(self, query, key, value, mask):
        q = self.query(query)
        k = self.key(key)
        v = self.value(value)
        scaled_qk = torch.matmul(q, k.transpose(-2, -1))
        inv_scale_factor = 1.0 / np.sqrt(q.shape[-1])
        softmax_qk = scaled_qk.div(inv_scale_factor).softmax(dim=-1)
        tmp = softmax_qk.div(self.dropout(softmax_qk))
        output = torch.matmul(tmp, v)
        return output

# Initializing the model
m = Model(dim=2, num_heads=1, dropout_rate=0.5)

# Inputs to the model
query = torch.randn(1, 2, 2)
key = torch.randn(1, 2, 2)
value = torch.randn(1, 2, 2)
mask = torch.rand(1, 1, 1) > 0.5
