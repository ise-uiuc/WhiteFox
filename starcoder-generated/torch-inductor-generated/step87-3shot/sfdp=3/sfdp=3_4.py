
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2, x3):
        v1 = torch.matmul(x1, x2.transpose(-2, -1))
        v2 = F.relu(v1)
        v3 = torch.matmul(v2, x3.transpose(-2, -1))
        v4 = F.softmax(v3, dim=-1)
        v5 = F.dropout(v4, p=0.2, training=True)
        v6 = torch.matmul(v5, x1)
        return v6

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(8, 5)
x2 = torch.randn(8, 5)
x3 = torch.randn(5, 8)
