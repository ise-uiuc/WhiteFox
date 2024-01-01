
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.selfattention = QueryKeyDotProductAttention()
 
    def forward(self, v1, v2, v3):
        v4 = self.selfattention(v1, v2, v3)
        return v4

# Initializing the model
m = Model()

# Inputs to the model
v1 = torch.randn(1, 2, 8, 8)
v2 = torch.randn(1, 2, 8, 8)
v3 = torch.randn(1, 2, 8, 8)
