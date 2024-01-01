
class Model(torch.nn.Module):
    def __init__(self, embed_dim, num_heads, dropout_p):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout_p = dropout_p
 
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
 
        self.linear_q = torch.nn.Linear(embed_dim, embed_dim)
        self.linear_k = torch.nn.Linear(embed_dim, embed_dim)
        self.linear_v = torch.nn.Linear(embed_dim, embed_dim)
 
        self.dropout = torch.nn.Dropout(dropout_p)
 
 
    def forward(self, query, key, value):
        q = self.linear_q(query)
        k = self.linear_k(key)
        v = self.linear_v(value)
 
        q = q.view(q.shape[0], q.shape[1], self.num_heads, self.head_dim).permute(0, 2, 1, 3) # B x num_heads x T x head_dim
        k = k.view(k.shape[0], k.shape[1], self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(v.shape[0], v.shape[1], self.num_heads, self.head_dim).permute(0, 2, 1, 3)
 
        scaled_qk = torch.matmul(q, k.transpose(-2, -1)) # B x num_heads x T x T
        scaled_qk = scaled_qk * (self.head_dim ** -0.5)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = self.dropout(softmax_qk)
        output = torch.matmul(dropout_qk, v).permute(0, 2, 1, 3).contiguous() # B x T x num_heads x head_dim
        return output.view(output.shape[0], output.shape[1], self.embed_dim)
 

# Initializing the model
m = Model(32, 8, 0.5)

# Inputs to the model
x1 = torch.randn(64, 8, 32)
x2 = torch.randn(64, 8, 32)
x2 = torch.randn(64, 8, 32)
