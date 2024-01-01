
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout_p = torch.nn.Parameter(torch.tensor(0.75))
        self.dropout = torch.nn.Dropout(self.dropout_p)
 
    def forward(self, x, y):
        v1 = torch.matmul(x, y.transpose(-2, -1))
        v2 = v1.div(self.inv_scale_factor)
        v3 = torch.nn.functional.softmax(v2, dim=-1)
        v4 = self.dropout(v3)
        v5 = torch.matmul(v4, y)
        return v5

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 4, 1)
y = torch.randn(4, 2, 1)
