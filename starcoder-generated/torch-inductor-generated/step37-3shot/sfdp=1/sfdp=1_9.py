
class Model(torch.nn.Module):
    def __init__(self, embedding_dim, num_heads, hidden_dim, dropout_p=0.1):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        if embedding_dim % num_heads!= 0:
            AssertionError("Please set model attributes properly!")
 
        self.qkv_proj = torch.nn.Linear(embedding_dim, hidden_dim * 3, bias=True)
        self.scaling_factor = torch.sqrt(torch.tensor(embedding_dim // num_heads, dtype=torch.float))
        self.pos_proj = torch.nn.Linear(embedding_dim, embedding_dim, bias=True)
        self.out_proj = torch.nn.Linear(embedding_dim, embedding_dim, bias=True)
        self.drop = torch.nn.Dropout(p=dropout_p)
 
        self.softmax = torch.nn.Softmax(dim=-1)
 
    def forward(self, x):
        qkv = self.qkv_proj(x)
        qkv = qkv.chunk(3, dim=-1)
        q, k, v = map(lambda t: t.contiguous().view(*t.size()[:-1], self.num_heads, self.hidden_dim).transpose(1, 2), qkv)
 
        qkv_scaled = q.matmul(k.transpose(-2, -1)) / self.scaling_factor
        attention_mask = generate_attention_mask(x, x, x)
        masker = qkv_scaled.transpose(1, 2).unsqueeze(3).repeat(1, 1, 1, v.size(1), 1)
        qkv_scaled_masked = qkv_scaled + torch.mul(attention_mask.permute(0, 2, 1), masker)
        qkv_softmaxed = self.softmax(qkv_scaled_masked).detach()
 
        qkv_dropped = self.drop(qkv_softmaxed)
        attn_out = qkv_dropped.matmul(v)
 
        out = attn_out.transpose(1, 2).contiguous().view(*attn_out.size()[:-2], self.embedding_dim)
        out = self.out_proj(out)
        return out

# Initializing the model
m = Model(embedding_dim, num_heads, hidden_dim, dropout_p)

# Inputs to the model
x = torch.randn(1, 5, 512)
