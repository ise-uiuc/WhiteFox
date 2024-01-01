
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = torch.nn.Dropout(dropout_p)
 
    def forward(self, x1, x2):
        v1 = torch.matmul(x1, x2.transpose(-2, -1))
        v2 = v1 * scale_factor
        v3 = torch.nn.functional.softmax(v2, dim=-1)
        v4 = self.dropout(v3)
        output = torch.matmul(v4, x2)
        return output

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 2, 3)
x2 = torch.randn(1, 3, 2)
