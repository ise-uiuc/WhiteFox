
def MultiHeadedAttention(h, d_model, dropout=0.1):
    def get_clones(module, N):
        return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
    assert d_model % h == 0
    d_k = d_model // h
    query = nn.Linear(d_model, d_model)
    key = nn.Linear(d_model, d_model)
    value = nn.Linear(d_model, d_model)
    dropout_layer = nn.Dropout(p=dropout)
    scale_factor = math.sqrt(d_k)
    output_layer = nn.Linear(d_model, d_model)
    return nn.ModuleList([nn.ModuleList([query, key, value]) for i in range(h)])

class Model(nn.Module):
    def __init__(h, d_model, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadedAttention(h, d_model, dropout)
        self.proj = nn.Linear(d_model, d_model)
 
    def get_clones(self.module, N):
        return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
 
    def forward(self, x0):
        for (query, key, value) in self.attention:
            query = query(x0)
            key = key(<KEY>
            value = value(x0)
            att = torch.matmul(query, key.transpose(-2, -1))
            att = att / scale_factor
            att = F.softmax(att, dim=-1)
            att = self.dropout_layer(att)
            x0 = torch.matmul(att, value)
        return self.proj(x0)

# Initializing the model
m = Model(h=8, d_model=64)

# Inputs to the model
x0 = torch.randn(1, 32, 64)
