
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout_p = 0.1
        self.dropout = torch.nn.Dropout(p=self.dropout_p)
 
    def forward(self, x1, x2):
        v1 = x1.matmul(x2.transpose(-2, -1))
        v2 = v1 / self.scale_factor
        v3 = torch.nn.functional.softmax(v2, dim=-1)
        v4 = self.dropout(v3)
        v5 = v4.matmul(x2)
        return v5

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(5, 6, 8)
x2 = torch.randn(6, 8, 7)
