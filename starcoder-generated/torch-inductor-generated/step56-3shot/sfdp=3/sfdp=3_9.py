
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale_factor = 1 / math.sqrt(64)
        self.dropout_p = 0.5
 
    def forward(self, x1, x2):
        v1 = torch.matmul(x1, x2.transpose(-2, -1))
        v2 = v1 * self.scale_factor
        v3 = v2.softmax(dim=-1)
        v4 = torch.nn.functional.dropout(v3, p=0.5)
        output = v4.matmul(v2)
        return output

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(8, 64)
x2 = torch.randn(8, 64)
