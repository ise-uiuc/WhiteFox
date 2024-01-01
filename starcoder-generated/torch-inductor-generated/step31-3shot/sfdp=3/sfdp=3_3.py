
class Model(torch.nn.Module):
    def __init__(self, scale_factor, dropout_p):
        super().__init__()
        self.scale_factor = scale_factor
        self.dropout_p = dropout_p
 
    def forward(self, x1, x2):
        v1 = torch.matmul(x1, x2.transpose(-2, -1))
        v2 = v1.mul(self.scale_factor)
        v3 = v2.softmax(dim=-1)
        v4 = torch.nn.functional.dropout(v3, p=self.dropout_p)
        v5 = torch.matmul(v4, x2)
        return v5

# Initializing the model
scale_factor = 1.0
dropout_p = 0.3
m = Model(scale_factor, dropout_p)

# Inputs to the model
x1 = torch.randn(10, 64)
x2 = torch.randn(20, 64)
