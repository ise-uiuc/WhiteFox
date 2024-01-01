
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.q = torch.nn.Linear(8, 8)
        self.k = torch.nn.Linear(8, 8)
        self.v = torch.nn.Linear(8, 8)
 
    def forward(self, q, k, v, scale_factor, dropout_p):
        qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = qk.mul(0.1)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=0.1)
        output = dropout_qk.matmul(v)
        return output

# Initializing the model
m = Model()

# Inputs to the model
q = torch.randn(8, 16, 8)
k = torch.randn(8, 16, 8)
v = torch.randn(8, 16, 8)
scale_factor = 0.1 * torch.ones((1, 16, 1), dtype=torch.float16)
dropout_p = 0.1
