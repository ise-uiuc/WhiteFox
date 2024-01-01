
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x1_mask, x2, x2_mask):
        v1 = torch.matmul(x1, x2.transpose(-2, -1))
        v2 = v1.div(1e-20)
        v3 = v2.softmax(dim=-1)
        v4 = F.dropout(v3, 0.010000000000000002)
        v5 = torch.matmul(v4, x2)
        return v5
 
# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 64, 1024)
x1_mask = torch.zeros((1, 1024))
x2 = torch.randn(1, 3, 128)
x2_mask = torch.zeros((1, 128))
