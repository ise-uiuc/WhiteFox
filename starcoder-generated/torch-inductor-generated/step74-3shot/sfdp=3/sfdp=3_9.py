
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax = torch.nn.Softmax(dim=-1)
 
    def forward(self, x1, x2, dropout_p):
        v1 = torch.matmul(x1, x2)
        scale_factor = (v1.size(-1)) ** -0.5
        v2 = v1.mul(scale_factor)
        v3 = self.softmax(v2)
        v4 = torch.nn.functional.dropout(v3, p=dropout_p)
        output = v4.matmul(x2)
        return output

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 4)
x2 = torch.randn(1, 4, 8) 
