
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.key = torch.rand(8, 16, 5)
        self.value = torch.rand(8, 16, 5)
        self.dropout_p = 0.1
 
    def forward(self, x1):
        v1 = torch.matmul(x1, self.key.transpose(-2, -1))
        v2 = v1.div(16.0)
        v3 = v2.softmax(dim=-1)
        v4 = torch.nn.functional.dropout(v3, self.dropout_p)
        v5 = v4.matmul(self.value)
        return v5

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(2, 16, 100, 5)
