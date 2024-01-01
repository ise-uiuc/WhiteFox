
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v, mask):
        qk = torch.matmul(q, k.transpose(-2, -1))
        scale = (query.size(-1) ** -0.5)
        scaled_lk = qk * scale
        softmax_lk = scaled_lk.softmax(dim=-1) * mask
        out = softmax_lk.matmul(v)

        return out

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(1, 16, 128, 8)
key =    torch.randn(1, 16, 256, 4)
value =  torch.randn(1, 16, 256, 4)
mask=    torch.ones(1, 1, 1, 144)
