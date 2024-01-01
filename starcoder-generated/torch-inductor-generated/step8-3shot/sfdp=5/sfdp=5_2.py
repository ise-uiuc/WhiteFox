
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x1, x2, x3):
        qk = x1 @ x2.transpose(-2, -1) / math.sqrt(x1.size(-1))
        qk = qk + x3
        attn_weight = torch.softmax(qk, dim=-1)
        attn_weight = torch.dropout(attn_weight, 0.3, True)
        output = attn_weight @ x2
        return output

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 5, 64)
x2 = torch.randn(1, 5, 64)
x3 = torch.randn(1, 5, 1)
