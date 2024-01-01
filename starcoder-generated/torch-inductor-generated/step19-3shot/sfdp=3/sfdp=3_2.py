
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dense = torch.nn.Linear(32, 32)
 
    def forward(self, q, v, k, scale_factor, dropout_p):
        q = self.dense(q)
        v = self.dense(v)
        k = self.dense(k)
        qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = qk.mul(scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(v)
        return output

# Initializing the model
m = Model()

# Inputs to the model
q = torch.randn(1, 7, 1, 128)
v = torch.randn(8, 1, 2, 64)
k = torch.randn(7, 8, 3, 64)
scale_factor = 10
dropout_p = 0.1
