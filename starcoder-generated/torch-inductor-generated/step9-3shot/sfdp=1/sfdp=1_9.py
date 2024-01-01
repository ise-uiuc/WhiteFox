
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q1, k1, v1):
        q_k = torch.matmul(q1, k1.transpose(-2, -1))
        s_q_k = q_k.div(0.5)
        softmax_q_k = s_q_k.softmax(dim=-1)
        dropout_q_k = torch.nn.functional.dropout(softmax_q_k, p=0.1)
        output = dropout_q_k.matmul(v1)
        return output

# Initializing the model
m = Model()

# Inputs to the model
q = torch.randn(1, 3, 100)
k = torch.randn(1, 3, 100)
v = torch.randn(1, 3, 100)
__output = m(q, k, v)

