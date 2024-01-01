
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, q, k):
        qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = qk.div(2)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=0.5)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model()

# Inputs to the model
q1 = torch.randn(1, 5, 2)
k1 = torch.randn(1, 3, 6)
value = torch.randn(1, 3, 5)
