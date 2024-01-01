
class Model(torch.nn.Module):
    def __init__(self, scale_factor):
        super().__init__()
        self.dropout_p = 0.1
        self.scale_factor = scale_factor
 
    def forward(self, x1, x2):
        v1 = torch.matmul(x1, x2.transpose(-2, -1))
        v2 = v1.div(self.scale_factor)
        v3 = v2.softmax(dim=-1)
        v4 = torch.nn.functional.dropout(v3, p=self.dropout_p)
        v5 = torch.matmul(v3, v2)
        return v5

# Initializing the model
scale_factor = 1
m = Model(scale_factor)

# Inputs to the model
x1 = torch.randn(1, 32, 23)
x2 = torch.randn(1, 512, 32)
