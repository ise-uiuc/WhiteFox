
class Model(torch.nn.Module):
    def __init__(self, w, b):
        super().__init__()
        self.linear = torch.nn.Linear(16, 8, True, w_init_fn=w, bias=b)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 * torch.clamp(v1 + 3, 0, 6)
        v3 = v2 / 6
        return v3

# Initializing parameters
p1 = torch.empty(8, 16)
torch.nn.init.normal_(p1, mean=0, std=0.1)
__output_name__ = p1
p2 = torch.empty(8)
torch.nn.init.uniform_(p2, a=0, b=1)
__output_name__ = p2

# Initializing the model
m = Model(p1, p2)

# Inputs to the model
x1 = torch.randn(1, 16)
