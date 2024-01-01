
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q1, k1, v1):
        qk1 = q1 @ k1.transpose(-2, -1) / math.sqrt(q1.size(-1))
        qk1 = qk1 + attn_mask
        attn_weight1 = torch.softmax(qk1, dim=-1)
        output1 = attn_weight1 @ v1
        return output1

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 2, 288, 288)
x2 = torch.randn(1, 2, 144, 144)
x3 = torch.randn(1, 2, 72, 72)
