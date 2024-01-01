
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.query = torch.nn.Linear(64, 64)
        self.key = torch.nn.Linear(64, 64)
        self.value = torch.nn.Linear(64, 64)

    def forward(self, x1):
        q = self.query(x1)
        k = self.key(x1)
        v = self.value(x1)
        qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = qk.div(4)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=0.8)
        output = dropout_qk.matmul(v)
        return output

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 64, 64)
