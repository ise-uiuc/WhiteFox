
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.query = torch.nn.Linear(4, 8)
        self.key = torch.nn.Linear(4, 8)
        self.value = torch.nn.Linear(4, 8)
 
    def forward(self, x1):
        qk = torch.matmul(self.query(x1), self.key(x1).transpose(-2, -1))
        scaled_qk = qk.div(1)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=0)
        output = dropout_qk.matmul(self.value(x1))
        return output

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 4)
