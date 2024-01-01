
class Model(torch.nn.Module):
    def __init__(self, dim, num_heads, dropout_p):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.dropout_p = dropout_p
        self.head_dim = self.dim // self.num_heads
        self.scaling = self.head_dim ** -0.5
        self.qkv = torch.nn.Linear(self.dim, self.dim * 3)
        self.dropout = torch.nn.Dropout(p=dropout_p)
        self.proj = torch.nn.Linear(self.dim, self.dim)
 
    def forward(self, inputs):
        qkv = self.qkv(inputs)
        qkv = qkv.reshape(qkv.shape[0], -1, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        scale = k.shape[-1] ** -0.5
        attn = torch.matmul(q * scale, k.transpose(-2, -1))
        attn = self.dropout(torch.nn.functional.softmax(attn, dim=-1))
        x = torch.matmul(attn, v)
        x = x.transpose(1, 2).reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3])
        x = self.proj(x)
        return x

# Initializing the model
m = Model(dim=512, num_heads=12, dropout_p=0)

# Inputs to the model
x1 = torch.randn(2, 4, 512)
