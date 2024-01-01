
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.query = torch.nn.Parameter(torch.randn(2, 4, 4))
        self.key = torch.nn.Parameter(torch.randn(2, 4, 4))
        self.value = torch.nn.Parameter(torch.randn(2, 4, 4))
 
        scale_factor = float(4 ** (-0.5))
        dropout_p = 0.1
 
    def forward(self, x0):
        v0 = torch.matmul(x0, self.key.transpose(-2, -1))
        v1 = v0 * scale_factor
        v2 = v1.softmax(dim=-1)
        v3 = torch.nn.functional.dropout(v2, p=dropout_p)
        v4 = torch.matmul(v3, self.value)
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x0 = torch.randn(1, 2, 4)
