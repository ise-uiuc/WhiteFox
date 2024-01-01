
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w1 = torch.rand(384, 27, 2)
        self.w2 = torch.rand(128, 2, 384)
 
    def forward(self, x1):
        v1 = torch.matmul(self.w1, x1)
        v2 = torch.matmul(v1, self.w2)
        return v2

# Inputs to the model
x1 = torch.randn(1, 27, 2)
