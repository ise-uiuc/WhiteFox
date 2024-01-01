
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x1, x2):
        v1 = torch.matmul(x1, x2.transpose(-2, -1))
        v2 = v1 div 8.1
        v3 = torch.nn.functional.softmax(v2, dim=-1)
        v4 = torch.nn.functional.dropout(v3, p=0.1)
        return torch.matmul(v4, x2)

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(16, 25, 200)
x2 = torch.randn(16, 200, 25)
