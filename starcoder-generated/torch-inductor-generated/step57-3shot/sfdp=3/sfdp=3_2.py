
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2, x3):
        qk = torch.matmul(x1, x2.transpose(-2, -1))
        scaled_qk = qk.mul(0.5)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=0.2)
        output = dropout_qk.matmul(x3)
        return output

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 64, 512)
x2 = torch.randn(1, 64, 512)
x3 = torch.randn(1, 512, 1024)
