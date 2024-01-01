
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.query = torch.nn.Linear(1000, 2000)
        self.key = torch.nn.Linear(1000, 2000)
        self.dropout_p = 0.1
        self.inv_scale_factor = math.sqrt((2000 // 2) / 0.1)
 
    def forward(self, x1, x2):
        q = self.query(x1)
        k = self.key(x2)
        qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = qk.div(self.inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p)
        dropup_qk = dropout_qk + torch.ones_like(dropout_qk)
        output = dropup_qk.matmul(value)
        return output

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(200, 1000)
x2 = torch.randn(200, 1000)
