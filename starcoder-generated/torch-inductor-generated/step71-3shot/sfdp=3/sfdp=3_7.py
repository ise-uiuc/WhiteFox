
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2):
        v1 = torch.matmul(x1, x2.transpose(-2, -1))
        return v1

# Initializing the model
m = Model()
 
# Inputs to the model
x1 = torch.randn(1, 768, 64)
x2 = torch.randn(1, 64, 768)
