
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.matmul1 = torch.matmul
 
    def forward(self, x1, x2):
        v1 = self.matmul1(x1, x2)
        v2 = v1.__div__(inv_scale_factor)
        v3 = torch.nn.functional.softmax(v2, dim=-1)
        dropout_v3 = torch.nn.functional.dropout(v3, p=dropout_p)
        v4 = dropout_v3.matmul(value)
        return output

# Initializing the model
m = Model()

# Inputs of the model
x1 = torch.randn(1, 16, 20)
x2 = torch.randn(1, 20, 40)

