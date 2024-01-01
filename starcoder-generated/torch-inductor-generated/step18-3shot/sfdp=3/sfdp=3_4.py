
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout_p = 0.0
        self.scale_factor = 1.0
    
    def forward(self, q, k, v):
        v1 = torch.matmul(q, k.transpose(-2, -1))
        v2 = v1 * self.scale_factor
        v3 = v2.softmax(dim=-1)
        v4 = torch.nn.functional.dropout(v3, p=self.dropout_p)
        v5 = torch.matmul(v4, v)
        return v5

# Initializing the model
m = Model()

# Inputs to the model
q = torch.randn(1, 16, 25)
k = torch.randn(1, 16, 49)
v = torch.randn(1, 16, 49)
