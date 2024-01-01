
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, q1, k1, v1):
        x1 = torch.matmul(q1, k1.transpose(-2, -1))
        x1 = x1 * self.scale_factor
        x2 = x1.softmax(dim=-1)
        x3 = torch.nn.functional.dropout(x2, p=self.dropout_p)
        x4 = x3.matmul(v1)
        return x4
 
# Initializing the model
dropout_p = 0.1
scale_factor = 2 ** 0.5
m = Model(dropout_p, scale_factor)

# Inputs to the model
q1 = torch.randn(1, 64, 1024)
k1 = torch.randn(1, 64, 1024)
v1 = torch.randn(1, 64, 1024)
