
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.m = torch.nn.Linear(2, 3)
 
    def forward(self, q, k, v):
        qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = qk.div(0.01)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=0.7)
        output = dropout_qk.matmul(v)
        return output

# Initializing the model
m = Model()

# Input to the model
q = torch.randn(4, 2)
k = torch.randn(4, 2)
v = torch.randn(4, 3)

