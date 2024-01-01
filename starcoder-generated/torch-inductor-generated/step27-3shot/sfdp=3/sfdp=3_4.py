
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.key = torch.nn.Linear(64, 64)
        self.query = torch.nn.Linear(64, 64)
        self.value = torch.nn.Linear(64, 64)
        self.scale_factor = torch.nn.Parameter(torch.empty(1))
        torch.nn.init.ones_(self.scale_factor)
 
    def forward(self, x1, x2, x3):
        qk = torch.matmul(self.query(x1), self.key(x2).transpose(-2, -1))
        scaled_qk = qk.mul(self.scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=x3)
        output = dropout_qk.matmul(self.value(x3))
        return output

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 64)
x2 = torch.randn(1, 64)
x3 = torch.randn(1, 64)
x_in = (x1, x2, x3)
