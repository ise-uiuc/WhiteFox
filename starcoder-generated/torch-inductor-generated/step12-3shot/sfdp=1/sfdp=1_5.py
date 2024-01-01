
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=0.2)
 
    def forward(self, x1, x2):
        v1 = torch.matmul(query, key.transpose(-2, -1))
        v2 = v1.div(inv_scale_factor)
        v3 = v2.softmax(dim=-1)
        v4 = self.dropout(v3)
        return v4.matmul(value)

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(60, 30, 100)
x2 = torch.randn(60, 30, 100)
