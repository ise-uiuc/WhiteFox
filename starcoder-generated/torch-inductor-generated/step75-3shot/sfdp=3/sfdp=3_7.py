
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale_factor = None
    def forward(self, q1, k1, v1):
        scale_factor = self.scale_factor
        if scale_factor is None:
            scale_factor = 1 / math.sqrt(q1.size(-1))
        v2 = torch.matmul(q1, k1.transpose(-2, -1))
        v3 = v2.mul(scale_factor)
        v4 = v3.softmax(dim=-1)
        v5 = torch.nn.functional.dropout(v4, p=0.0)
        v6 = torch.matmul(v5, v1)
        return v6

# Initializing the model
m = Model()

# Inputs to the model
q1 = torch.randn(1, 2, 4)
k1 = torch.randn(2, 4, 16)
