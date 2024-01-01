
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2, x3, x4):
        qk = x1 @ x2.transpose(-2, -1) / math.sqrt(x1.size(-1))
        qk = qk + x3
        attn_weight = torch.softmax(qk, dim=-1)
        output = attn_weight @ x4
        return output

# Initializing the model
m = Model()

# Initializing the inputs and attention mask
x1 = torch.randn(2, 6, 16)
x2 = torch.randn(2, 6, 16)
x3 = torch.randn(2, 6, 6, 6)
x4 = torch.randn(2, 6, 16)
