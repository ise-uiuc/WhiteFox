
class SelfAttentionLayer(nn.Module):
    def __init__(self, dim, dropout_p=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout_p)
        self.softnorm = nn.Softmax(dim=-1)
        self.fc_1 = nn.Linear(dim, dim)
        self.fc_2 = nn.Linear(dim, dim)

    def forward(self, x, mask):
        t1 = x.matmul(self.fc_1.weight.transpose(-2, -1))  # (B, LQ, H)->(B, H, LQ)
        t2 = x.matmul(self.fc_2.weight.transpose(-2, -1))  # (B, LQ, H)->(B, H, LC)
        # tensor(B, LQ, H), (B, LC, H)
        t3 = t1 + t2
        t3 = t3.masked_fill_(mask, -float('inf'))
        attn_scores = self.softnorm(t3)  # B * LQ * H
        if self.training:
            x = x + self.dropout(attn_scores)
        else:
            x = x + attn_scores

        return x

# Initializing the model
model = SelfAttentionLayer()

# Inputs to the model
x = torch.rand(2, 6, 5) # (B, LQ, H)
mask = torch.triu(x.new(x.size(1), x.size(1)).fill_(1), diagonal=1) == 0
