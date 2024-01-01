
class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0., bias=True):
        from math import sqrt
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        head_dim = embed_dim // num_heads
        self.scale = head_dim**-0.5
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=bias)
        self.dropout = nn.Dropout(dropout)
 
    def forward(self, x):
        q, k, v = self.qkv(x).reshape(x.shape[0], x.shape[1], 3, self.num_heads, self.embed_dim // self.num_heads).permute(2, 0, 3, 1, 4).contiguous().chunk(3, dim=0)
        scaled = q * self.scale
        softmax_qk = F.softmax(scaled, dim=-1)
        dropout_qk = self.dropout(softmax_qk)
        return torch.einsum("ndhe,ndse->nhds", dropout_qk, v).reshape(x.shape[0], self.num_heads, x.shape[1], self.embed_dim // self.num_heads).permute(0,2,1,3).contiguous().reshape(x.shape[0], x.shape[1], self.embed_dim)

# Initializing the model
# Please modify the `256` and `12` with expected (based on your model) values to pass the test
m = MultiheadAttention(256, 12)

# Inputs to the model
# Please modify the `32` and `196` with expected (based on your model) values to pass the test
x1 = torch.randn(16, 32, 196)
