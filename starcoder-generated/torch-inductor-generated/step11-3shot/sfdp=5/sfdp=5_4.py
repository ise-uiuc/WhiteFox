
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, query, key, value, attn_mask, dropout_p):
        qk = query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))
        qk = qk + attn_mask
        attn_weight = torch.softmax(qk, dim=-1)
        attn_weight = torch.dropout(attn_weight, dropout_p, True)
        output = attn_weight @ value
        return output

# Initializing the model, input and attention mask
query = torch.randn(1, 2, 3, 4)
key = torch.randn(1, 2, 6, 4)
value = torch.randn(1, 2, 6, 5)
attn_mask = torch.randn(1, 2, 2, 4)
dropout_p = 0.5
