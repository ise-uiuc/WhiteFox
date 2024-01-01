
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2, x3):
        v1 = x1 @ x2.transpose(-2, -1)
        v1 /= math.sqrt(v1.size(-1))
        v2 = v1 + x3
        v2_max = torch.max(v1, dim=1, keepdim=True).values
        v2_max = v2_max.expand_as(v2) - v2_max

        attn_mask = (v2_max + v2.le(0).to(v2.dtype)).detach()

        attn_weight = torch.softmax(attn_mask, dim=-1)
        output = attn_weight @ x3

        return output

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 128, 64, 128)
x2 = torch.randn(37, 128, 128)
x3 = torch.randn(37, 128)
