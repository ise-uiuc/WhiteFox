
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attn_mask = torch.ones(7, 7, dtype=torch.bool).cuda(0)
 
    def forward(self, x1, x2):
        qk = x1 @ x2.transpose(-2, -1) / math.sqrt(x1.size(-1))
        qk = qk + self.attn_mask
        attn_weight = torch.softmax(qk, dim=-1)
        output = attn_weight @ x1
        return output

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 7, 10)
x2 = torch.randn(1, 2, 5)
