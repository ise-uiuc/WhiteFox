
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.query = torch.nn.Parameter(torch.randn(2, 4, 3))
        self.key = torch.nn.Parameter(torch.randn(6, 4, 5))
 
    def forward(self, v1):
        qk = self.query @ self.key.transpose(-2, -1) / math.sqrt(self.query.size(-1))
        attn_mask = torch.zeros(qk.size(), dtype=torch.bool, device=qk.device)
        attn_mask[:, :, :3, 1:4] = True
        v5 = qk + attn_mask
        attn_weight = torch.softmax(qk, dim=-1)
        v7 = attn_weight @ value
        return v7

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(2, 4, 5)
__output_y1__ = m(x1)

