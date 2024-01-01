
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
  
    def forward(self, x14):
        v7 = x14.transpose(-2, -1)
        v8 = x14 @ v7
        v9 = v8 / math.sqrt(x14.size(-1))
        v10 = v9 + x9
        v11 = torch.softmax(v10, dim=-1)
        v12 = x14 @ v11
        return v12

# Initializing the model
m = Model()

# Inputs to the model
x14 = torch.randn(1, 64, 64)
