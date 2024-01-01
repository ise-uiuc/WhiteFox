
class Model(torch.nn.Module):
    def __init__(self, *, n_heads, head_size):
        super().__init__()
        self.n_heads = n_heads
        self.head_size = head_size
        self.scale_factor = head_size ** -0.5
 
    def forward(self, query, key, value, dropout_p):
        shape = query.size()[:-1]
        qk = query.view(shape + (self.n_heads, self.head_size)).matmul(key.transpose(-2, -1).contiguous().view(shape + (self.n_heads, self.head_size)).transpose(-1, -2))
        qk = qk.mul(self.scale_factor)
        softmax_qk = qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        return dropout_qk.matmul(value.view(shape + (self.n_heads, self.head_size)).permute((0, 2, 1, 3))).view(shape + (self.n_heads * self.head_size,))

# Initializing the model
m = Model(n_heads=8, head_size=16)

# Inputs to the model
query = torch.randn(1, 3, 64, 64)
key = torch.randn(1, 3, 64, 64)
value = torch.randn(1, 3, 64, 64)
dropout_p = 0.9
