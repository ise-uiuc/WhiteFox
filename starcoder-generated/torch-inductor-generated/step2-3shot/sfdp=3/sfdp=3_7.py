
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2):
        v1 = torch.matmul(x1, x2.transpose(-2, -1))
        v2 = v1 * torch.scalar_tensor(0.5)
        v3 = v2.softmax(-1)
        v4 = torch.nn.functional.dropout(v3, p = torch.scalar_tensor(0.2))
        return torch.matmul(v4, x2)

# Initializing the model
m = Model()

# Inputs for the model
x1 = torch.randn(3, 16, 128)
x2 = torch.randn(3, 128, 16)
