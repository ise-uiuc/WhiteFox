
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.q_linear = torch.nn.Linear(8*3, 64)
        self.k_linear = torch.nn.Linear(64, 32)
 
    def forward(self, x1):
        b = x1.shape[0]
        x1x1 = x1.view([b, 8*3])
        q = self.q_linear(x1x1)
        k = self.k_linear(x1x1)
        attn_map = (q @ k.transpose(-2, -1)) / math.sqrt(q.size(-1))
        attn_mask = torch.ones([1,32,32])
        x1x1 = attn_map + attn_mask
        x2 = torch.softmax(x1x1, dim=-1)
        x3 = (x1x1 @ k.transpose(-2, -1)).transpose(-2, -1)
        return x3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 64*3)
