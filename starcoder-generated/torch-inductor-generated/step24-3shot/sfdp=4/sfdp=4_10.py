
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2):
        qk = x1 @ x2.transpose(-2, -1) / math.sqrt(x1.size(-1))
        qk = qk + attn_mask
        attn_weight = torch.softmax(qk, dim=-1)
        output = x3 @ attn_weight
        return output

# Initializing the model
m = Model()

# Initializing inputs
x1 = torch.randn(4,8, 32, 32)
x2 = torch.randn(4,8, 32, 32)
