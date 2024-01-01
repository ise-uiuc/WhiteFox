
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.inv_scale_factor = 10
 
    def forward(self, x1, x2):
        v1 = torch.matmul(x1, x2.transpose(-2, -1))
        v2 = v1.div(self.inv_scale_factor)
        v3 = v2.softmax(dim=-1)
        v4 = torch.nn.functional.dropout(v3, p=0.75)
        v5 = v4.matmul(x2)
        return v5

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(4, 2, 3, 3)
x2 = torch.randn(4, 2, 5, 3)
