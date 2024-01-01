
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x1, min_value=-2, max_value=3):
        v1 = torch.nn.functional.linear(x1, torch.nn.init.normal_(torch.zeros(inp_dim, out_dim)))
        v2 = torch.clamp_min(v1, min_value)
        v3 = torch.clamp_max(v2, max_value)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, inp_dim)
