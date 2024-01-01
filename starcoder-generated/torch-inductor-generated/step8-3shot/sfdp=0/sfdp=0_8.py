
class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super().__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 3)
        self.dropout = nn.Dropout(p=dropout)
        self.scale = math.sqrt(d_model)
 
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linears, (query, key, value))]
        x, attn = attention(query, key, value, mask=self.dropout(mask), scale=self.scale)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
        return self.linears[-1](x)
 
class Model(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadedAttention(h, d_model, dropout=dropout)
        self.projection = nn.Linear(d_model, d_model)
 
    def forward(self, x, mask=None):
        return self.projection(self.attention(x, x, x, mask=mask))
 
