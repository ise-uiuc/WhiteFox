
class Model(torch.nn.Module):
    def __init__(self, heads, q_dim, k_dim, v_dim):
        super().__init__()
        self.heads = heads
        self.q_dim = q_dim
        self.k_dim = k_dim
        self.v_dim = v_dim
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
 
    def forward(self, query, key, value, mask=0):
        q = self.conv(query)
        v = self.conv(value)
        k = self.conv(key)
        a = (q @ k.transpose(-2, -1) / math.sqrt(self.k_dim)) + attn_mask
        attn_weight = torch.softmax(a, dim=-1)
        output = attn_weight @ v
        return output

# Initializing the model
m = Model(heads=8, q_dim=256, k_dim=128, v_dim=128)

# Inputs to the model
query = torch.randn(1, 3, 64, 64)
key = torch.randn(1, 3, 128, 128)
value = torch.randn(1, 3, 128, 128)
attn_mask = torch.randn(1, 1, 1, 256)
