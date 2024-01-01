
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2, scale_factor=1.0, dropout_p=0.5):
        y1 = torch.matmul(x1, x2.transpose(-2, -1))
        y2 = y1 * scale_factor
        y3 = y2.softmax(dim=-1)
        y4 = torch.nn.functional.dropout(y3, p=dropout_p)
        y5 = torch.matmul(y4, x2)
        return y5

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
x2 = torch.randn(1, 3, 64, 64)
