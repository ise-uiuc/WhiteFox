
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = torch.nn.Dropout()
 
    def forward(self, x1, x2):
        v1 = x1 @ x2.transpose(-2, -1)
        scale = 1/math.sqrt(x1.shape[-1])
        v2 = v1 * scale
        v3 = v2.softmax(-1)
        v4 = self.dropout(v3)
        o = v4 @ x3
        return o

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(16, 8, 8)
x2 = torch.randn(16, 8, 8)
