
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = torch.nn.MultiheadAttention(128, 8, 0.1)
 
    def forward(self, x1):
        q = k = self.attn.in_proj_q_weight
        v = self.attn.in_proj_v_weight
        out, _ = self.attn(q, k, v)
        return out

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 20, 128)
