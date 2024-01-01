
class Model(torch.nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.proj_query = torch.nn.Linear(embed_dim, embed_dim)
        self.proj_key = torch.nn.Linear(embed_dim, embed_dim)
        self.proj_value = torch.nn.Linear(embed_dim, embed_dim)
 
    def forward(self, x1):
        q = self.proj_query(x1)
        k = self.proj_key(x1)
        v = self.proj_value(x1)
        # print(q.shape, k.shape, v.shape)
        q, k, v = q.reshape(1, -1, self.num_heads, self.head_dim), k.reshape(1, -1, self.num_heads, self.head_dim), v.reshape(1, -1, self.num_heads, self.head_dim)
 
