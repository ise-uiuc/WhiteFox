
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2):
        v1 = torch.matmul(x1, x2.transpose(-2, -1))
        v2 = v1 * 0.5
        v3 = v2.softmax(dim=-1)
        v4 = torch.nn.functional.dropout(v3, p=0.1)
        return v4.matmul(x2)

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(1, 3, 8, 16)
key = torch.randn(1, 3, 8, 20)
