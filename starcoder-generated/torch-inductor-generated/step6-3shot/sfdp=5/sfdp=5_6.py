
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        n = 12
        d_q = 8
        d_k = 16
        d_v = 8
        d_ff = 16
        d_emb = 256
        n_head = 4
        dropout_p = 0.2

    def forward(self, query, key, value):
        v1 = query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))
        v1 = v1 + self.attn_mask
        v2 = torch.softmax(qkv, -1)
        v2 = torch.dropout(v2, self.dropout, True)
        v3 = v2 @ value
        return v3

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(8, 12, 8)
key = torch.randn(8, 12, 16)
value = torch.randn(8, 12, 8)
