
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.matmul_qk = torch.nn.Linear(8, 8)
        self.dropout_qk = torch.nn.Dropout(dropout_p)
 
    def forward(self, q, k):
        qk = self.matmul_qk(q)
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = self.dropout_qk(softmax_qk)
        output = dropout_qk.matmul(v)
        return output

# Initializing the model
m = Model()

# Inputs to the model
q = torch.randn(1, 8)
k = torch.randn(1, 8)
v = torch.randn(1, 8)
