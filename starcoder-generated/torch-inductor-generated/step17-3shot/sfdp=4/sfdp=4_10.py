
class MultiheadAttentionIdentity(torch.nn.MultiheadAttention):
    def forward(self, query, key, value, attn_mask=None):
        return super().forward(query, key, value, attn_mask)
 
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = MultiheadAttentionIdentity(32, 8)
 
    def forward(self, x1, x2):
        v1 = self.attn(x1, x2, x2)
        return v1

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(128, 32, 64)
x2 = torch.randn(128, 32, 64)
