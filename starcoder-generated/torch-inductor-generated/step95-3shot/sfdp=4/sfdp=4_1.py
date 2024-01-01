
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.q1 = torch.nn.Parameter(torch.randn(3, 2))
        self.k5 = torch.nn.Parameter(torch.randn(3, 2))
        self.v2 = torch.nn.Parameter(torch.randn(3, 2))
        self.mask = torch.nn.Parameter(mask.squeeze(0))
    def forward(self):
        qk = self.q1 @ self.k5.transpose(-2, -1) / math.sqrt(self.q1.size(-1))
        qk = qk + self.mask
        attn_weight = torch.softmax(qk, dim=-1)
        output = attn_weight @ self.v2
        return output
# Inputs to the model
