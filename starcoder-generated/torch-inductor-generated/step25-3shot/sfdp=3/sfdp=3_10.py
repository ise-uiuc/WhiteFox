
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale_factor = 1.0 / math.sqrt(512)
        self.dropout = torch.nn.Dropout(p=0.1)
 
    def forward(self, x1, x2):
        v1 = torch.matmul(x1, x2.transpose(-2, -1))
        v2 = v1 * self.scale_factor
        v3 = v2.softmax(dim=-1)
        v4 = self.dropout(v3)
        return v4.matmul(x2)

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(512, 128, 512)
x2 = torch.randn(512, 512, 128)
