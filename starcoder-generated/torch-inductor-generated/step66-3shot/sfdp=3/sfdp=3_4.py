
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = torch.nn.Dropout(0.1)
 
    def forward(self, x1):
        v1 = torch.matmul(x1, x1.transpose(-2, -1))
        v2 = v1 * 1.95163 # Scale the dot product by a fixed factor
        v3 = v2.softmax(dim=-1)
        v4 = self.dropout(v3)
        v5 = v4.matmul(x1)
        return v5

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 10, 512, 512)
