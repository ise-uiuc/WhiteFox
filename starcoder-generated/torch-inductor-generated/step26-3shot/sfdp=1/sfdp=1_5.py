
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.query = torch.nn.Linear(768, 1)
        self.key = torch.nn.Linear(768, 1)
        self.value = torch.nn.Linear(768, 1)
 
    def forward(self, x1):
        q = self.query(x1)
        k = self.key(x1)
        v = self.value(x1)
 
        qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = qk.div(0.00440160646)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=0.1)
        output = dropout_qk.matmul(v)
 
# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 768, 25)
