
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, q1, k1, v1):
        qk = torch.matmul(q1, k1.transpose(-2, -1))
        scaled_qk = qk.div(2 ** 0.5)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=0.2)
        output = dropout_qk.matmul(v1)
        return output

# Initializing the model
m = Model()

# Inputs to the model
q1 = torch.randn(1, 8, 32, 32)
k1 = torch.randn(1, 8, 32, 32)
v1 = torch.randn(1, 8, 32, 32)
output = m(q1, k1, v1)

