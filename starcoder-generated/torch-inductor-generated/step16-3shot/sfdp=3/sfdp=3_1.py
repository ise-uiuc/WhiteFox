
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=0.5)
 
    def forward(self, q, k, v, scale):
        dropout_qk = self.dropout(torch.softmax(torch.matmul(q, k.transpose(-2, -1)) * scale, dim=-1))
        return torch.matmul(dropout_qk, v)

# Initializing the model
m = Model()

# Inputs to the model
q = torch.randn(64, 25, 2 * 3085)
k = torch.randn(64, 25, 2 * 3085)
v = torch.randn(64, 25, 2 * 3085)
scale = 1.0
