
class Model(torch.nn.Module):
    def __init__(self, dim=2, scale_factor=1.0, dropout_p=0.0):
        super().__init__()
        self.scale_factor = scale_factor
        self.softmax = torch.nn.Softmax(dim=dim)
        self.dropout = torch.nn.Dropout(p=dropout_p)
 
    def forward(self, x1, x2):
        v1 = torch.matmul(x1, x2.transpose(-2, -1))
        v2 = v1 * self.scale_factor
        v3 = self.softmax(v2)
        v4 = self.dropout(v3)
        v5 = torch.matmul(v4, x2)
        return v5

# Initializing the model
m = Model(dim=-1)

# Inputs to the model
x1 = torch.randn(1, 2, 3)
x2 = torch.randn(5, 2, 4)
