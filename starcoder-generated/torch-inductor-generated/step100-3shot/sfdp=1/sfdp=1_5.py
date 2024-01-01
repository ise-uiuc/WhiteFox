
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2, x3, x4):
        qk = torch.matmul(x1, x2.transpose(-2, -1))
        scaled_qk = qk.div(10.)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dp_qk = torch.nn.functional.dropout(softmax_qk, p=0.1)
        return torch.matmul(dp_qk, x3)

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 100, 2)
x2 = torch.randn(1, 2, 100)
x3 = torch.randn(1, 100, 50)
x4 = torch.randn(1, 100, 30)
