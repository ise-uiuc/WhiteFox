
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1):
        v1 = torch.matmul(x1, x1.transpose(-2, -1))
        v2 = v1.div(1e-15)
        v3 = torch.nn.functional.softmax(v2, dim=3)
        v4 = torch.nn.functional.dropout(v3, p=0.5)
        return torch.matmul(v4, x1)

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 10, 128)
