
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.v_1 = nn.Linear(100, 128)

    def forward(self, q, k, mask):
        v1 = self.v_1(q)
        v2 = k.transpose(1, 2)
        v2 = v2 + v1
        v3 = v2 / SQRT_D
        v4 = torch.softmax(v3, dim=-1)
        k = v2 * v4
        k = k.transpose(2, 3)
        v1 = k * v
        return v4

# Initializing the model
m = Model()

# Inputs to the model
q = torch.randn(1, 4, 100)
k = torch.randn(1, 6, 100)
mask = torch.randn(1, 6, 4, 6)
