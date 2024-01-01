
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.key = torch.nn.Linear(8, 8)
        self.query = torch.nn.Linear(8, 8)
        self.value = torch.nn.Linear(8, 8)
 
    def forward(self, q1, k1, v1):
        qk = torch.matmul(q1, k1.transpose(-2, -1))
        inv_scale_factor = math.sqrt(64)
        scaled_qk = qk.div(inv_scale_factor)
        dropout_p = 0.1
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(scaled_qk, p=dropout_p)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model()

# Inputs to the model
q1 = torch.randn(1, 8)
k1 = torch.randn(1, 8)
v1 = torch.randn(1, 8)
