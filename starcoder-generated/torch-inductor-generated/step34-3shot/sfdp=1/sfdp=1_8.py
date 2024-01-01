
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, q, k, v, scale_factor, dropout_p):
        v1 = torch.matmul(q, k.transpose(-2, -1))
        v2 = v1.div(scale_factor)
        v3 = v2.softmax(dim=-1)
        v4 = torch.nn.functional.dropout(v3, p=dropout_p)
        v5 = torch.matmul(v4, v)
 
        return v5
 
# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 1, 1024)
y = torch.randn(1, 1, 1024)
scale_factor = 10.0
dropout_p = 0.5
