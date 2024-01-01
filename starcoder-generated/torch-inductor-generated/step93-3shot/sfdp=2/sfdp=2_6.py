
class Model(torch.nn.Module):
    def __init__(self, attention_dim, num_heads, input_dim, dropout_p):
        super().__init__()
        self.attention_dim = attention_dim
        self.num_heads = num_heads
        self.input_dim = input_dim
        self.dropout_p = dropout_p
 
        self.query = torch.nn.Linear(input_dim, attention_dim * num_heads)
        self.key = torch.nn.Linear(input_dim, attention_dim * num_heads)
        self.value = torch.nn.Linear(input_dim, attention_dim * num_heads)
        self.out = torch.nn.Linear(attention_dim * num_heads, input_dim)
 
    def forward(self, query, key, value, mask=None):
        q = self.query(query)
        k = self.key(key)
        v = self.value(value)
        if self.num_heads == 1:
            q, k, v = torch.chunk(q, self.num_heads, dim=-2), \
                torch.chunk(k, self.num_heads, dim=-2), \
                torch.chunk(v, self.num_heads, dim=-2)
 
 
        q, k, v = map(lambda x: x.contiguous().view(*x.size()[:-2], x.size(-2), self.num_heads, x.size(-1) // self.num_heads), (q, k, v))
        m = k.transpose(-2, -1).matmul(q).div(self.attention_dim**-0.5)
 
        if mask is not None:
            m += mask
 
        m = torch.nn.functional.dropout(m, self.dropout_p)
 
        a = m.matmul(v).view(*m.size()[:-2], -1)
        o = self.out(a)
 
        return o

# Initializing the model
m = Model(attention_dim=16, num_heads=2, input_dim=256, dropout_p=0.03)

# Inputs to the model
query = torch.randn(1, 256, 1, 1)
key = torch.randn(1, 256, 1, 1)
value = torch.randn(1, 256, 1, 1)
