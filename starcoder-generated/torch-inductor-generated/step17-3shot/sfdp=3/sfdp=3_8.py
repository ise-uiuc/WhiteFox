
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = torch.nn.Dropout(0.75)
 
    def forward(self, x1, x2):
        v1 = torch.matmul(x1, x2.transpose(-2, -1)) 
        v2 = v1 * 0.5
        v3 = v2.softmax(dim=-1)
        v4 = self.dropout(v3)
        output = v4.matmul(x2)
        return output

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 128, 512)
x2 = torch.randn(1, 128, 560)
