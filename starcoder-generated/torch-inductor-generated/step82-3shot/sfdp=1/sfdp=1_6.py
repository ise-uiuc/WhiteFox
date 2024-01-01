
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=0.1)
 
    def forward(self, q, k, v, scale_factor, dropout_p):
        v1 = torch.matmul(q, k.transpose(-2, -1))
        v2 = v1 / scale_factor
        v3 = torch.nn.functional.softmax(v2, dim=-1)
        v4 = self.dropout(v3)
        return torch.matmul(v4, v)

# Initializing the model
m = Model()

# Inputs to the model
q = torch.randn(1, 16, 8)
k = torch.randn(1, 16, 4)
v = torch.randn(1, 16, 4)
scale_factor = torch.randint(1, 256)
