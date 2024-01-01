
class Model(torch.nn.Module):
    def __init__(self, scale_factor, dropout_p):
        super().__init__()
        self.dropout_p = dropout_p
        self.scale_factor = torch.nn.Parameter(scale_factor)
 
    def forward(self, x1, x2, x3):
        v1 = torch.matmul(x1, x2.transpose(-2, -1))
        v2 = v1 * self.scale_factor
        v3 = v2.softmax(dim=-1)
        v4 = torch.nn.functional.dropout(v3, p=self.dropout_p)
        v5 = torch.matmul(v4, x3)
        return v5

# Initializing the model with a scale factor of 0.1 and a dropout probability of 0.1
m = Model(scale_factor=0.1, dropout_p=0.1)

# Inputs to the model
x1 = torch.randn(1, 6, 3, 3)
x2 = torch.randn(1, 6, 3, 3)
x3 = torch.randn(1, 3, 4)
