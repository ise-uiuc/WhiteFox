
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.query = torch.nn.Linear(32, 16)
        self.key = torch.nn.Linear(32, 16)
        self.value = torch.nn.Linear(32, 16)

    def forward(self, q, k, v, attn_mask):
        qk = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))
        qk = qk + attn_mask
        attn_weight = torch.softmax(qk, dim=-1)
        output = attn_weight @ v
        return output

# Initializing the model
m = Model()

# Inputs to the model
q = torch.randn(4, 3, 16)
k = torch.randn(4, 10, 16)
v = torch.randn(4, 10, 16)
attn_mask = torch.FloatTensor([[[[0, -10000.0, 0], [-10000.0, 0, 0], [0, 0, 0]]]]).repeat(4, 1, 1)
