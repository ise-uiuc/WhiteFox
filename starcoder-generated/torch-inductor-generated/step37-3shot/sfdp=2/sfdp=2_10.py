
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
 
    def forward(self, x1, x2):
        qk = torch.matmul(x1, x2.transpose(-2, -1))
        scaled_qk = qk.div(4.0)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, 0.25)
        v1 = dropout_qk.matmul(x1)
        return v1

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(3, 5, 64)
x2 = torch.randn(3, 9, 64)
