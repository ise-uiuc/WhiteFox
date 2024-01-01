
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2, x2_mask):
        v1 = torch.matmul(x1, x2.transpose(-2, -1))
        v2 = v1 / math.sqrt(v1.shape[-1])
        v3 = v2.softmax(dim=-1)
        v4 = torch.nn.functional.dropout(v3, p=0.3)
        v5 = torch.matmul(v4, x2)
        return v5

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(64, 512, 16)
x2 = torch.randn(64, 16, 1024)
x2_mask = torch.randn(1, 1, 1, 1024)
