
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x1, x2):
        qk = self.matmul(x1, x2.transpose(-2, -1))
        scaled_qk = qk.mul(scale)
        softmax_qk = scaled_qk.softmax(-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        return dropout_qk.matmul(x2)

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 1, 128)
x2 = torch.randn(1, 1, 128)
