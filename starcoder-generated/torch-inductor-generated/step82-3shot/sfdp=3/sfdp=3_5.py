
class Model(nn.Module):
    def __init__(self, embed_dim, num_heads, scale_factor, dropout_p):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale_factor = scale_factor
        self.dropout_p = dropout_p

        self.k_proj = nn.Linear(embedding_dim, embedding_dim=num_heads * head_dim)
        self.v_proj = nn.Linear(embedding_dim, embedding_dim=num_heads * head_dim)
        self.q_proj = nn.Linear(embedding_dim, embedding_dim=num_heads * head_dim)
        self.out_proj = nn.Linear(num_heads * head_dim, embedding_dim)
    
    def forward(self, query, key, value):
        q, k, v = self.q_proj(query), self.k_proj(key), self.v_proj(value)
        attn = torch.matmul(q, k.transpose(-2, -1))
        attn = attn * self.scale_factor
        attn = nn.functional.softmax(attn, dim=-1)
        attn = nn.functional.dropout(attn, p=self.dropout_p)
        proj = self.out_proj(attn)
        return proj

# Initializing the model
m = Model(embed_dim=256, num_heads=3, scale_factor=0.5, dropout_p=0.2)

# Inputs to the model
query = torch.randn(256, 256) 
key = torch.randn(256, 256) 
value = torch.randn(256, 256) 
