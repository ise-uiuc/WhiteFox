
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale_factor = 4
        self.dropout_p = 0.5
 
    def forward(self, x1, x2):
        v1 = x1.matmul(x2.transpose(-2, -1))
        v2 = v1 * self.scale_factor
        v3 = torch.nn.functional.softmax(v2)
        v4 = torch.nn.functional.dropout(v3)
        v5 = v4.matmul(x2)
        return v5

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(3, 4)
x2 = torch.randn(4, 5)
