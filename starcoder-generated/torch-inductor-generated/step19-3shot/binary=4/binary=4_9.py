
class Model(torch.nn.Module):
    def __init__(self, size_w, size_b):
        super().__init__()
        self.linear = torch.nn.Linear(size_w, size_b)
 
    def forward(self, x1, x2):
        v1 = self.linear(x1)
        v2 = v1 + x2
        return v2
 
size_w = 64
size_b = 32
 
m = Model(size_w, size_b)

# Inputs to the model
x1 = torch.randn(2, size_w)
x2 = torch.randn(2, size_b)
