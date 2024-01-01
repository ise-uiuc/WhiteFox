
class Model(torch.nn.Module):
    def __init__(self, d_model, nhead, nhid, dropout_p, d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_model, nhid)
        self.attn = nn.MultiheadAttention(nhid, nhead, dropout_p)
        self.linear3 = nn.Linear(d_model, nhid)
        self.linear4 = nn.Linear(nhid, d_model)
    
    def forward(self, x):
        out, _ = self.attn(self.linear1(x), self.linear1(x), self.linear3(x), attn_mask = None)
        return self.linear4(out)

# Initializing the model
m = Model(d_model, nhead, nhid, dropout_p, d_ff)
x = torch.randn(query.size(0), query.size(1), query.size(2))
