
class Model(torch.nn.Module):
    def __init__(self, num_heads=1, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.dropout = dropout

    def forward(self, query, key, value, mask=None):
        score = torch.matmul(query, key.transpose(-2, -1))
        score.mul_(1 / np.sqrt(self.num_heads))
        if mask is not None:
            score.masked_fill_(mask, -1e9)
        p_attn = F.softmax(score, dim=-1)
        p_attn = F.dropout(p_attn, p=self.dropout)
        output = torch.matmul(p_attn, value)
        return output

# Inputs to the model
key = torch.randn(2, 1, 5, 3)
value = torch.randn(10, 1, 3)
query = torch.randn(2, 1, 4, 3)
mask = torch.tensor([[8., -1e9, -1e9], [-1e9, 1., -1e9]])
