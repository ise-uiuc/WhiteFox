
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = torch.nn.Dropout(dropout_p)
 
    def forward(self, x, y):
        a = torch.matmul(x, y.transpose(-2, -1))
        b = a / inv_scale_factor
        c = torch.nn.functional.dropout(b.softmax(dim=-1), self.dropout_p)
        d = torch.matmul(c, z)
        return d

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 3, hidden_size)
y = torch.randn(1, 3, hidden_size)
