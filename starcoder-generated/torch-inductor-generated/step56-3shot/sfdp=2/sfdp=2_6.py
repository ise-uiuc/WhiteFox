
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2):
        qk = torch.matmul(x1, x2.t())
        scaled_qk = qk.div(1e-10)
        softmax_qk = F.softmax(scaled_qk, dim=-1)
        dropout_qk = F.dropout(softmax_qk, p=0.1)
        output = torch.matmul(dropout_qk, x1)
        return output

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(8, 3)
x2 = torch.randn(10, 8)
