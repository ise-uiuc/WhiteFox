
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Linear(8, 8)
        self.gru = torch.nn.GRU(8, 8)
 
    def forward(self, x, h, attn_mask):
        query = self.conv(x)
        key = self.conv(h)
        value = torch.zeros_like(h)
        attn_weights = torch.softmax((query @ key.transpose(-1, -2) / math.sqrt(query.size(-1))), dim=-1)
        attn_weights = attn_mask.float() * attn_weights + (1 - attn_mask.float()) * -1e4
        context_vector = attn_weights @ value
        return context_vector

# Initializing the model
m = Model()

# Input
x = torch.randn(6, 4, 8)
h = torch.randn(1, 6, 8)
attn_mask = torch.randint(2, (6, 6))

# Inputs to the model
with __device__:
    