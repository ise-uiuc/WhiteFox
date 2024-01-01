
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)
 
    def forward(self, x1):
        q = x1.T
        k = x1.T
        inv_scale_factor = 0.7071067811865476
        dropout_p = 0.1    
        qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        y = torch.matmul(dropout_qk, self.linear(x1))
        return y

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 1)
