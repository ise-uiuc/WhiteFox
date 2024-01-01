
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale_factor = 1 / (64 ** 0.5)
 
    def forward(self, x1, x2):
        v1 = torch.matmul(x1, x2.transpose(-2, -1))
        v2 = v1 * self.scale_factor
        v3 = torch.nn.functional.softmax(v2, dim=-1)
        v4 = torch.nn.functional.dropout(v3, p=0.7)
        v5 = torch.matmul(v3, x2)
        return v5

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 64, 6, 32)
x2 = torch.randn(1, 64, 32, 4)
