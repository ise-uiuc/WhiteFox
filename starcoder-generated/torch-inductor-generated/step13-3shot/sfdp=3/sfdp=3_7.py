
class Model(torch.nn.Module):
    def __init__(self, scale_factor, dropout_p):
        super().__init__()
        self.scale_factor = scale_factor
        self.dropout_p = dropout_p
 
    def forward(self, x1, x2):
        v1 = torch.matmul(x1, x2.transpose(-2, -1))
        v2 = v1.mul(self.scale_factor)
        v3 = torch.nn.functional.softmax(v2, dim=-1)
        v4 = torch.nn.functional.dropout(v3, p=self.dropout_p)
        return v4.matmul(x2)

# Initializing the model
m = Model(1e-05, 0.3)

# Inputs to the model
x1 = torch.randn(1, 8, 16)
x2 = torch.randn(1, 8, 24)
