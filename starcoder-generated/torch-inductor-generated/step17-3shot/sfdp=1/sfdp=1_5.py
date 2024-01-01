
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, q, k, v, isf, dp):
        qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = qk.div(isf)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dp)
        output = dropout_qk.matmul(v)
        return output

# Initializing the model
m = Model()

# Inputs to the model. isf is the inverse scale factor, dp is dropout probability
q1 = torch.randn(3, 5, 6)
k1 = torch.randn(5, 4, 6)
v1 = torch.randn(5, 4, 6)
isf1 = torch.tensor(1.0)
dp1 = torch.tensor(0.0)
