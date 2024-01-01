
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(selfq1, k1, v1):
        qk = torch.matmul(q1, k1.transpose(-2, -1))
        scaled_qk = qk.div(16)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(v1)
        return output

# Initializing the model
m = Model()

# Inputs to the model
q1 = torch.randn(4, 100, 64)
k1 = torch.randn(4, 200, 64)
v1 = torch.randn(4, 200, 512)
