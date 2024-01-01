
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.query = torch.nn.Linear(128, 128)
        self.key = torch.nn.Linear(128, 128)
        self.value = torch.nn.Linear(128, 128)
 
    def forward(self, x1, x2):
        qk = torch.matmul(x1, self.key(x2).transpose(-2, -1))
        scaled_qk = qk.div(2 ** 0.5)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=0.8)
        return self.value(x2).matmul(dropout_qk.transpose(-2, -1))
    
# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(4, 128)
x2 = torch.randn(4, 128)
