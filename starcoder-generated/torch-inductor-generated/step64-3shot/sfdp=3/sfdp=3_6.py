
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.query = torch.nn.Linear(3, 8)
        self.key = torch.nn.Linear(3, 8)
        self.value = torch.nn.Linear(3, 8)

    def forward(self, x1):
        qk = torch.matmul(self.query(x1).transpose(-2, -1), self.key(x1).transpose(-2, -1))
        scaled_qk = qk * 0.0625
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=0.4)
        return self.value(x1).matmul(dropout_qk)

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 16, 3)
