
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.inv_scale_factor = 10.0
        self.dropout_p = 0.2
 
    def forward(self, x1, x2, x3):
        v1 = torch.matmul(x1, x2.transpose(-2, -1))
        v2 = v1.div(self.inv_scale_factor)
        v3 = F.softmax(v2, dim=-1)
        v4 = F.dropout(v3, p=self.dropout_p)
        return torch.matmul(v4, x3)

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 5, 200)
x2 = torch.randn(1, 5, 200)
x3 = torch.randn(1, 200, 100)
