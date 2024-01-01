
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = torch.nn.Dropout(dropout_p)
 
    def forward(self, input_tensor, x1, x2):
        v1 = torch.matmul(x1, x2.transpose(-2, -1))
        v2 = v1 * scale_factor
        v3 = v2.softmax(dim=-1)
        v4 = self.dropout(v3)
        v5 = v4.matmul(x3)
        return v5

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(5, 128)
x2 = torch.randn(5, 128)
x3 = torch.randn(5, 128)
