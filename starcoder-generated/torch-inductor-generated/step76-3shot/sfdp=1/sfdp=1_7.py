
class Model(torch.nn.Module):
    def __init__(self, dim, heads, dropout):
        super().__init__()
        self.scale = dim ** -0.5
        self.dropout = dropout
        self.qkv = torch.nn.Linear(dim, dim*3)
        self.softmax = torch.nn.Softmax(dim=-1)
 
    def forward(self, x):
        q, k, v = torch.chunk(self.qkv(x), 3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=heads), (q, k, v))
        q = q * self.scale
        qk = torch.matmul(q, k.transpose(-2, -1))
        softmax_qk = self.softmax(qk)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout)
        output = torch.matmul(dropout_qk, v)
        output = rearrange(output, 'b h n d -> b n (h d)')
        return output

# Initializing the model
m = Model(dim=64, heads=8, dropout=0.1)

# Input to the model
x = torch.randn(1, 64, 1024)
