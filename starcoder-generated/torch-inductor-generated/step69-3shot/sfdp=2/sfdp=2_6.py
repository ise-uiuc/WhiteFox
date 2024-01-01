
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, q, k, v):
        qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = qk.div(0.1)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=0.2)
        output = dropout_qk.matmul(v)
        return output
# Initializing the model
m = Model()
# Inputs to the model
q = torch.randn(1, 1, 7)
k = torch.randn(1, 1, 10)
v = torch.randn(1, 1, 10)
