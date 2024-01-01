
class Model(torch.nn.Module):
    def __init__(self, attention_dim, batch_size=16, num_heads=4, dropout_p=0.2):
        super().__init__()
        self.query = torch.nn.Linear(128, num_heads * attention_dim)  # A query tensor
        self.key = torch.nn.Linear(128, num_heads * attention_dim)  # A key tensor 
        self.value = torch.nn.Linear(128, num_heads * attention_dim)  # A value tensor

    def forward(self, q, k, v, mask=None):
        num_heads, attention_dim = self.query.out_features, self.query.in_features // self.num_heads
        scale_factor = 1 / math.sqrt(attention_dim)
        q = self.query(q)
        q = q.reshape(size=(-1, batch_size * num_heads, attention_dim))
        k = self.key(k)
        k = k.reshape(size=(-1, batch_size * num_heads, attention_dim))
        v = self.value(v)
        v = v.reshape(size=(-1, batch_size * num_heads, attention_dim))
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk * scale_factor
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model(attention_dim=128, num_heads=5, dropout_p=0.2)

# Inputs to the model
batch_size, length, num = 32, 12, 30
q = torch.randn(length, batch_size, 128)
k = torch.randn(num, batch_size, 128) 
v = torch.randn(num, batch_size, 128)
mask = (torch.from_numpy(np.random.randint(0, 2, (length, num))).bool().cuda())
