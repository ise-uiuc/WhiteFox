
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2):
        mul1 = torch.matmul(x1, x2.transpose(-2, -1))
        mul2 = mul1 / math.sqrt(x1.size(-1))
        add1 = mul2 + 0.2
        attn_weight = torch.softmax(add1, dim=-1)
        out = torch.matmul(attn_weight, x2)
        return out, attn_weight

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 5, 3)
x2 = torch.randn(1, 5, 2)
__output__, __attention_weight__ = m(x1, x2)