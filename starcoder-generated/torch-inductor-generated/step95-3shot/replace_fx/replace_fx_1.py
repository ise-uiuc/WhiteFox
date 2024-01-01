
class Model(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        x2 = torch.nn.functional.dropout(x1)
        x3 = torch.mean(x1)
        x3 = (x3, x1)
        x4 = torch.rand_like(x3[1])
        x5 = torch.rand_like(x3).values() # Dict
        x6 = torch.rand_like(x3).values()[1] # Tensor
        return x5
# Inputs to the model
x1 = torch.randn(1, 2, 2)
