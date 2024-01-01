
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.query = torch.nn.Linear(512, 256, bias=False)
        self.key = torch.nn.Linear(512, 256, bias=False)
        self.value = torch.nn.Linear(512, 256, bias=False)
 
    def forward(self, x1, x2):
        qk = torch.matmul(self.query(x1), self.key(x2).transpose(-2, -1))
        scaled_qk = qk.div((512 ** -0.2))
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=0.8)
        output = dropout_qk.matmul(self.value(x2))
        return output

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 512)
x2 = torch.randn(1, 512)
