
class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.proj_q = nn.Linear(embed_dim, embed_dim)
        self.proj_k = nn.Linear(embed_dim, embed_dim)
        self.proj_v = nn.Linear(embed_dim, embed_dim)
        self.drop_mha_out = nn.Dropout(self.dropout)
        self.fc = nn.Linear(self.embed_dim, self.embed_dim)
 
    def forward(self, query, key, value, attn_mask=None, output_attentions=False):
        residual = query
        batch_size, tgt_len, embed_dim = query.size()
        head_dim = embed_dim // self.num_heads
        scale_factor = 1 / math.sqrt(head_dim)
        q = self.proj_q(query).view(batch_size, tgt_len, self.num_heads, head_dim).permute(0, 2, 1, 3)
        k = self.proj_k(key).view(batch_size, -1, self.num_heads, head_dim).transpose(1, 2)
        v = self.proj_v(value).view(batch_size, -1, self.num_heads, head_dim).transpose(1, 2)
        q, k, v = q * scale_factor, k * scale_factor, v * scale_factor
        scores = torch.matmul(q, k)
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0).unsqueeze(0).bool()
            scores = scores.masked_fill(attn_mask, score_mask_value)
        scores = F.softmax(scores, dim=-1)
        scores = torch.nn.functional.dropout(scores, self.dropout, training=self.training)
        attn = torch.matmul(scores, v).permute(0, 2, 1, 3)
        attn = attn.contiguous().view(batch_size, tgt_len, embed_dim)
        attn = self.drop_mha_out(attn)
        return self.fc(attn), scores

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
        self.dropout = 0.5
        self.mha = MultiheadAttention(embed_dim=256, num_heads=8, dropout=self.dropout)
 
    def forward(self, x1, x2):
        v1, _ = self.mha(x1, x2, x2, torch.ones(8, 8, dtype=torch.bool, device=x1.device))
        return v1

model = Model()

# Inputs for the model
x1 = torch.randn(1, 8, 256)
x2 = torch.randn(1, 8, 256)
