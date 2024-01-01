
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax = torch.nn.Softmax(dim=-1)
 
    def forward(self, q, k, v, p):
        q_k = torch.matmul(q, k.transpose(-2, -1))
        s_q_k = q_k.div(k.size()[-1]**0.5)
        softmax_q_k = self.softmax(s_q_k)
        d_q_k = torch.nn.functional.dropout(softmax_q_k, p)
        output = d_q_k.matmul(v)
        return output

# Initializing the model
m = Model()

# Generating test input
q = torch.randn(16, 32, 80, 3)
k = torch.randn(16, 40, 67, 3)
v = torch.randn(16, 40, 67, 3)
p = 0.05

