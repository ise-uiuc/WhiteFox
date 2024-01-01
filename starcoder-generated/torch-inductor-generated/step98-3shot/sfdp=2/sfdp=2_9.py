
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2):
        v1 = torch.matmul(x1, x2.transpose(-2, -1))
        inv_scale_factor = torch.tensor(1/10)
        v2 = v1.div(inv_scale_factor)
        v3 = v2.softmax(dim=-1)
        dropout_p = torch.tensor(0.0)
        v4 = torch.nn.functional.dropout(v3, dropout_p)
        v5 = torch.matmul(v4, x2)
        return v5

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 10, 256)
x2 = torch.randn(1, 10, 256)
