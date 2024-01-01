
class Model(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.scale_factor = (in_dim // 2) ** (-0.5)
 
    def forward(self, x1, x2):
        v1 = torch.matmul(x1, x2.transpose(-2, -1))
        v2 = v1.div(self.scale_factor)
        v3 = v2.softmax(dim=-1)
        v4 = torch.nn.functional.dropout(v3)
        v5 = torch.matmul(v4, x2)
        return v5

# Initializing the model
m = Model(in_dim=3, out_dim=4)

# Inputs to the model
x1 = torch.randn(4, 3)
x2 = torch.randn(5, 3, 4)
