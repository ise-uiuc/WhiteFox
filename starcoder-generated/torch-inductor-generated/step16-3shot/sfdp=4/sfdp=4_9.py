
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attn_mask = nn.Parameter(torch.randn([1, 1, 1, 64], dtype=torch.float32))
        self.attn_q = torch.nn.Linear(8, 8)
        self.attn_k = torch.nn.Linear(8, 8)
        self.attn_v = torch.nn.Linear(8, 8)
 
    def forward(self, x1, x2):
        q = self.attn_q(x1)
        k = self.attn_k(x2)
        v = self.attn_v(x2)
        qk = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))
        qk = qk + self.attn_mask
        attn_weight = torch.softmax(qk, dim=-1)
        attn_output = attn_weight @ v
        return attn_output

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 1, 8)
x2 = torch.randn(1, 64, 8)
