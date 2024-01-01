
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.query = torch.nn.Parameter(torch.rand(2048, 1024))
        self.key = torch.nn.Parameter(torch.rand(1024, 1024))
        self.value = torch.nn.Parameter(torch.rand(1024, 1024))
        self.dropout_p = 0.1
        self.scale_factor = 1024 ** -.5
 
    def forward(self, x1):
        v1 = torch.matmul(x2, self.key.transpose(-2, -1))
        v2 = v1.div(self.scale_factor)
        v3 = v2.softmax(dim=-1)
        v4 = torch.nn.functional.dropout(v3, p=self.dropout_p)
        v5 = self.value.matmul(v4)
        return v5

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 2048, 1024)
x2 = torch.randn(1, 2048, 1024)
