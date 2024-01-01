
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale_factor = 1
 
    def forward(self, q, k, v):
        v1 = torch.matmul(q, k.transpose(-2, -1))
        v2 = v1 * self.scale_factor
        v3 = v2.softmax(dim=-1)
        v4 = torch.nn.functional.dropout(v3, p=0.2)
        v5 = torch.matmul(v4, v)
        return v5

# Initializing the model
m = Model()

# Inputs to the model
scale = 10
q = torch.randn(1, 10, 16)
