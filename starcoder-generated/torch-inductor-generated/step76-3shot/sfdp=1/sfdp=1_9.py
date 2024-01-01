
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.query_key = torch.nn.Linear(3, 8, bias=False)
        self.dropout = torch.nn.Dropout(p=0.1)
        self.value = torch.nn.Linear(8, 16, bias=False)
 
    def forward(self, x1, x2):
        q1 = self.query_key(x1)
        k2 = self.query_key(x2)
        qk = torch.matmul(q1, k2.transpose(-2, -1))
        inv_scale_factor = (3. ** 0.5) / 5.
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = self.dropout(softmax_qk)
        v3 = dropout_qk.matmul(self.value)
        return v3


# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
x2 = torch.randn(1, 3, 64, 64)
