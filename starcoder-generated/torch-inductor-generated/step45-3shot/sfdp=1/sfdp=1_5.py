
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x, y):
        v1 = torch.matmul(x, y.transpose(-2, -1))
        v2 = v1.div(0.125)
        v3 = v2.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(v3, p=0.10)
        output = torch.matmul(dropout_qk, y)
        return output

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 16, 128)
y = torch.randn(1, 128, 2)
