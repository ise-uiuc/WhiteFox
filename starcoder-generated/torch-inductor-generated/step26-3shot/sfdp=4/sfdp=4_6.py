
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.q = torch.nn.Parameter(torch.randn(8, 8, 4, 64, 64))
        self.pos = PosEncoding(8, 4, 64, 64)
        self.k = self.q
        self.v = self.q

    def forward(self, l1):
        l1 = self.pos(l1)
        q = self.q
        k = self.k
        v = self.v
        attn_mask = -1e4 * (torch.triu(torch.ones(4, 4, 64, 64), diagonal=1)
                   + torch.tril(torch.ones(4, 4, 64, 64)))
        qk = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))
        qk = qk + attn_mask
        attn_weight = torch.softmax(qk, dim=-1)
        output = attn_weight @ v
        return output, attn_weight, attn_mask

# Initializing the model
m = Model()

# Inputs for the model
l1 = torch.randn(4, 8, 64, 64)
