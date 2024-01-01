
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.MultiheadAttention(8, 1, dropout=0.0)
 
    def forward(self, x1, x2):
        v1, v2 = self.model(x1, x2, x2)
        return v1

# Initializing the model
m = Model()

# Input to the model
x1 = torch.randn(1, 8, 10)
x2 = torch.randn(1, 8, 10)
