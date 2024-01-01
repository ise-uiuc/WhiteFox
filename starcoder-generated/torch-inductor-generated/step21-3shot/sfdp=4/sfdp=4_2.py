
class Model(torch.nn.Module):
    def __init__(self,  embed_dims, num_heads, qkv_bias, qk_scale, dropout_ratio):
        super().__init__()
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.scale = qk_scale or self.embed_dims ** -0.5
 
        self.qkv = torch.nn.Linear(in_features=embed_dims, out_features=embed_dims * 3, bias=qkv_bias)
        self.attn_drop = torch.nn.Dropout(dropout_ratio)
        self.proj = torch.nn.Linear(in_features=embed_dims, out_features=embed_dims)
        self.proj_drop = torch.nn.Dropout(dropout_ratio)
 
    def forward(self, qkv):
        q, k, v = torch.chunk(self.qkv(qkv), 3, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(qkv.shape)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

# Initializing the model
m = Model(embed_dims=64, num_heads=4, qkv_bias=False, qk_scale=None, dropout_ratio=0.1)

# Inputs to the model
qkv = torch.randn(4, 128, 64)
