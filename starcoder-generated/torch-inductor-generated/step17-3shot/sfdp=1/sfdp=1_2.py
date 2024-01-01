
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.key = torch.nn.Linear(200, 50)
        self.value = torch.nn.Linear(200, 50)
        self.query = torch.nn.Linear(200, 50)
 
    def forward(self, q, k, v, inv_scale_factor, dropout_p):
        qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(v)
        return output

# Initializing the model
m = Model()

# Inputs to the model
q = torch.randn(20, 50, 200)
k = torch.randn(20, 13, 200)
v = torch.randn(20, 13, 200)
inv_scale_factor = 1 + torch.randn(1, 1, 1)
dropout_p = 0.5
