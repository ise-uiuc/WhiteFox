
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale_factor = 1
        self.dropout_p = 0.3
 
    def forward(self, x1, x2):
        v1 = torch.matmul(x1, x2.transpose(-1, -2))
        v2 = v1.mul(self.scale_factor)
        v3 = v2.softmax(dim=-1)
        v4 = torch.nn.functional.dropout(v3, p=self.dropout_p)
        v5 = torch.matmul(v4, x2)
        return v5

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 10, 16)
x2 = torch.randn(1, 5, 16)
