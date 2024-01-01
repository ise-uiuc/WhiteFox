
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, q1, k2, v3, inv_scale_factor, dropout_p):
        qk1 = torch.matmul(q1, k2.transpose(-2, -1))
        scaled_qk1 = qk1.div(inv_scale_factor)
        softmax_qk1 = scaled_qk1.softmax(-1)
        dropout_qk1 = torch.nn.functional.dropout(softmax_qk1, p=dropout_p)
        output = dropout_qk1.matmul(v3)
        return output

# Initializing the model
m = Model()

# Inputs to the model
q1 = torch.randn(1, 1, 64)
k2 = torch.randn(1, 7, 64)
v3 = torch.randn(1, 7, 64)
inv_scale_factor = torch.tensor(3.14)
dropout_p = torch.tensor(0.5)
