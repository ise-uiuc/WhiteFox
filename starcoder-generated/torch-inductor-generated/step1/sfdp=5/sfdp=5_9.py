
class Model(torch.nn.Module):
    def __init__(self, num_heads, max_length, dropout_p):
        super().__init__()
        self.num_heads = num_heads
        self.max_length = max_length
        self.dropout_p = dropout_p
        weights = np.random.rand(num_heads, max_length, max_length)
        self.weights = torch.FloatTensor(weights)
        dropout = np.random.rand(max_length, max_length)
        self.dropout = torch.FloatTensor(dropout)

    def forward(self, q, k, v):
       attn_mask = (q.abs() == float('inf')).to("cuda")
       qk = torch.matmul(q, k.transpose(-2, -1))
       qk = qk / np.sqrt(self.weights.size(-1))
       qk = qk + attn_mask
       attn = F.softmax(qk, dim=-1)
       attn = F.dropout(attn, self.dropout_p, training=True)
       output = attn @ v
       return output

# Initializing the model
num_heads = 8
m = Model(num_heads, 64, 0.075)
x = torch.randn(1, 64, 64)
