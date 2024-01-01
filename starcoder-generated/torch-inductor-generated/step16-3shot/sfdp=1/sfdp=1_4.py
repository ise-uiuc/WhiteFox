
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout_p = 0.1
 
    def forward(self, x1, x2):
        v1 = x1.matmul(x2.transpose(-2, -1))
        scale = 1 / math.sqrt(v1.size(-1))
        v2 = v1 * scale
        v3 = torch.nn.functional.softmax(v2, dim=-1)
        v4 = torch.nn.functional.dropout(v3, p=self.dropout_p)
        return x2.matmul(v4)

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 64, 512)
x2 = torch.randn(1, 512, 64)
