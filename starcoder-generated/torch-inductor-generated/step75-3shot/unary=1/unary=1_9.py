
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x1):
        v1 = torch.nn.functional.linear(x1, in_features=16, out_features=8, bias=False)
        v2 = v1 * 0.5
        v3 = v1 * v1 * v1 * 0.044715
        v4 = v3 * 0.7978845608028654
        v5 = torch.tanh(v4)
        v6 = v5 + 1
        v7 = v2 * v6
        return v7
  
# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(512, 16)
