
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x1, x2):
        v1 = torch.Tensor.matmul(x1,x2)
        v2 = torch.add(v1,x2)
        v3 = F.relu(v2)
        return v3

# Initializing the model 
m = Model()

# Inputs to the model
x1 = torch.randn(1, 20)
x2 = torch.randn(1, 20)
