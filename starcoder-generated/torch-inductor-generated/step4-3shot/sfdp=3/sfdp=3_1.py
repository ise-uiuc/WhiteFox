
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = torch.nn.LayerNorm([100, 10])
 
    def forward(self, input):
        v1 = self.norm(input)
        v2 = torch.nn.functional.softmax(v1, dim=-1)
        v3 = torch.nn.functional.dropout(v2, p=0.5, training=self.training)
        v4 = v3.matmul(input)
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 100, 10)
