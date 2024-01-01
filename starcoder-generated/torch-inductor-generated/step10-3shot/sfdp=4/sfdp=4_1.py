
class Model(torch.nn.Module):
    def __init__(self, dim_in, dim_hidden):
        super().__init__()
        self.multi_head_attn = torch.nn.MultiheadAttention(dim_in, 8)

    def forward(self, x1, x2, x3):
        v1 = self.multi_head_attn(x1, x2, x3, need_weights=False)
        return v1

# Initializing the model
m = Model(32, 32)

# Inputs to the model
x1 = torch.randn(1, 8, 32, 32)
x2 = torch.randn(1, 8, 32, 32)
x3 = torch.randn(1, 8, 32, 32)
