
class Model1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.q = torch.nn.Linear(128, 128)
        self.k = torch.nn.Linear(128, 128)
        self.v = torch.nn.Linear(128, 128)
        self.scale = math.sqrt(128)
        
    def forward(self, q, k, v, attn_mask):
        qk = q @ k.transpose(-2, -1) /self.scale
        qk = qk + attn_mask
        attn_weight = torch.softmax(qk, dim=-1)
        output = attn_weight @ v
        return output

# Initializing the model
m = Model1()

# Inputs to the model
x1 = torch.randn(1, 128)
x2 = torch.randn(1, 128)
x3 = torch.randn(1, 128)
x4 = torch.randn(1, 1, 128, 128)
