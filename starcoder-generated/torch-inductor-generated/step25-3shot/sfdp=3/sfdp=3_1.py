
class Model(torch.nn.Module):
    def __init__(self, scale_factor=None, dropout_p=None):
        super().__init__()
        self.scale_factor = scale_factor
        self.dropout_p = dropout_p
        self.linear = torch.nn.Linear(6, 1)
 
    def forward(self, q, k, v):
        mul = q.matmul(k.transpose(1, 2))
        mul = mul / self.scale_factor
        softmax = nn.functional.softmax(mul, dim=-1)
        drop = nn.functional.dropout(softmax, p=self.dropout_p)
        out = drop.matmul(v)
        return out

# Initializing the model
scale_factor = 0.125
dropout_p = 0.6
m = Model(scale_factor, dropout_p)

# Inputs to the model
q = torch.randn(5, 3, 6)
k = torch.randn(5, 12, 6)
v = torch.randn(5, 12, 6)
