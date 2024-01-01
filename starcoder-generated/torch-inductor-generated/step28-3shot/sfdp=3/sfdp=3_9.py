
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = torch.nn.Dropout(0.5)
 
    def forward(self, x1, x2, x3):
        scale_factor = 1.0 / ((x1.size(1) * x2.size(1) * x3.size(1)) ** 0.5)
        qk = torch.matmul(x1, x2.transpose(-2, -1))
        scaled_qk = qk.mul(scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = self.dropout(softmax_qk)
        x = dropout_qk.matmul(x3)
        return x

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 64, 10)
x2 = torch.randn(1, 64, 10)
x3 = torch.randn(1, 10, 8)
