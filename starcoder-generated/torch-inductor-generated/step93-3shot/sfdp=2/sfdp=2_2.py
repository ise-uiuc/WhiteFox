
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale_factor = torch.nn.Parameter(torch.ones(1) * 1.0)
 
    def forward(self, x1, x2, x3):
        y1 = torch.matmul(x1, x2.transpose(-2, -1))
        y2 = y1.div(self.scale_factor)
        y3 = y2.softmax(dim=-1)
        y4 = torch.nn.functional.dropout(y3, p=0.2)
        output = y4.matmul(x3)
        return output

# Initializing the model
m = Model()

# Inputs to the model
x_query = torch.randn(1, 4, 16)
x_key = torch.randn(1, 8, 16)
x_value = torch.randn(1, 8, 32)
