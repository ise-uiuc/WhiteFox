
class Model(torch.nn.Module):
    def __init__(self, dropout_p, scale_factor):
        super().__init__()
        self.dropout_p = dropout_p
        self.scale_factor = scale_factor
 
    def forward(self, x1, x2, x3):
        v1 = torch.matmul(x1, x2.transpose(-2, -1))
        v2 = v1.div(self.scale_factor)
        v3 = v2.softmax(dim=-1)
        v4 = torch.nn.functional.dropout(v3, self.dropout_p)
        v5 = torch.matmul(v4, x3)
        return v5

# Initializing the model
m = Model(0.2, 0.2)

# Inputs to the model
x1 = torch.randn(1, 1, 128)
x2 = torch.randn(1, 1, 128)
x3 = torch.randn(1, 1, 128)
