
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.key = torch.nn.Linear(120, 120)
        self.query = torch.nn.Linear(40, 120)
        self.value = torch.nn.Linear(800, 800)
        self.scale_factor = 40.0 / math.sqrt(120)
    
    def forward(self, x1, x2, x4):
        qk = torch.matmul(self.query(x1), self.key(x2).transpose(-2, -1))
        scaled_qk = qk.mul(self.scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=0.05)
        output = dropout_qk.matmul(self.value(x4))
        return output

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(120, 40)
x2 = torch.randn(120, 120)
x4 = torch.randn(800, 800)
