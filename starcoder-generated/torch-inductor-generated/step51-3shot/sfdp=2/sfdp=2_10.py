
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout_p = 1.0
        self.dropout = torch.nn.Dropout(self.dropout_p)
 
    def forward(self, x1, x2, x3):
        v1 = torch.matmul(x1, torch.transpose(x2, -2, -1))
        v2 = v1.div(0.01)
        v3 = torch.nn.functional.softmax(v2, dim=-1)
        v4 = self.dropout(v3)
        v5 = torch.matmul(v4, x3)
        return v5

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(2, 5, 3)
x2 = torch.randn(2, 3, 6)
x3 = torch.randn(2, 6, 8)
