
class Model(torch.nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.attn_head_dim = embed_dim // num_heads
        self.all_head_size = self.attn_head_dim * num_heads
        self.query = torch.nn.Linear(self.embed_dim, self.all_head_size, bias=True)
        self.key = torch.nn.Linear(self.embed_dim, self.all_head_size, bias=True)
        self.value = torch.nn.Linear(self.embed_dim, self.all_head_size, bias=True)
        self.attn_dropout = torch.nn.Dropout(0.0)
        self.proj = torch.nn.Linear(self.all_head_size, self.embed_dim)
        self.proj_dropout = torch.nn.Dropout(0.0)
 
    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_heads, self.attn_head_dim)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)
 
    def forward(self, x1, x2):
        query = self.transpose_for_scores(self.query(x1))
        key = self.transpose_for_scores(self.key(x2))
        value = self.transpose_for_scores(self.value(x2))
        mixed_qk = torch.matmul(query, key.transpose(-2, -1))
        scale = self.attn_head_dim ** -0.5
        attn_qk = mixed_qk * scale
        attn_qk = torch.nn.functional.softmax(attn_qk, dim=-1)
        attn_qk = self.attn_dropout(attn_qk)
        output = attn_qk.matmul(value)
        output = output.transpose(1, 2)
        concat = output.reshape(output.size(0), -1)
        proj = self.proj(concat)
        proj = self.proj_dropout(proj)
        return proj

# Initializing the model
m = Model(embed_dim=50, num_heads=5)

# Inputs to the model
x1 = torch.randn(20, 50)
x2 = torch.randn(20, 50)
