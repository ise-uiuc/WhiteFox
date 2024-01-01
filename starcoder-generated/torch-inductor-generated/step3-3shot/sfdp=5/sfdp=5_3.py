
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = math.sqrt(8)

    def forward(self, q1, k1, v1, attn_mask):
        q2 = q1 @ k1.transpose(-2, -1)
        q2 = q2 / (self.scale)
        q2 = q2 + attn_mask
        attn_weight = torch.softmax(q2, dim=-1)
        attn_weight = torch.dropout(attn_weight, dropout_p, True)
        output = attn_weight @ v1
        return output

# Initializing the model
m = Model()

# Inputs to the model
q1 = torch.randn(1, 3, 64, 64)
k1 = torch.randn(1, 3, 8, 8)
v1 = torch.randn(1, 3, 8, 8)
attn_mask = torch.ones(1, 64, 64, dtype=torch.long)
