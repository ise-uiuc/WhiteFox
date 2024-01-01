
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale_factor = 16

    def forward(self, x1, x2):
        v1 = torch.matmul(x1, x2.transpose(-2, -1))
        v2 = v1.div(self.scale_factor)
        v3 = v2.softmax(dim=-1)
        v4 = torch.nn.functional.dropout(v3, p=0.5)
        v5 = torch.matmul(v4, x2)
        return v5

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(256, 64, 256)
x3 = torch.randn(256, 32, 64)
