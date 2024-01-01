
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2):
        v1 = torch.matmul(x1, x2)
        v2 = torch.nn.functional.dropout(v1)
        v3 = v2.softmax(dim=-1)
        v4 = torch.matmul(v3, v1)
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(128, 60, 512)
x2 = torch.randn(128, 512, 3)
