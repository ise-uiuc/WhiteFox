
class Model(torch.nn.Module):
    def __init__(self, dropout_p, scale_factor):
        super().__init__()
        self.dropout_p = dropout_p
        self.scale_factor = scale_factor
 
    def forward(self, x1, x2):
        v1 = torch.matmul(x1, x2.transpose(-2, -1))
        v2 = v1.div(self.scale_factor)
        v3 = torch.nn.functional.softmax(v2, dim=-1)
        v4 = torch.nn.functional.dropout(v3, p=self.dropout_p)
        v5 = torch.matmul(v4, x2)
        return v5
 
dropout_p = 0.125
scale_factor = 1.5368709838867188e-05
m = Model(dropout_p = dropout_p, scale_factor = scale_factor)

# Inputs to the model
x1 = torch.randn(1, 3, 256)
x2 = torch.randn(1, 64, 3)
