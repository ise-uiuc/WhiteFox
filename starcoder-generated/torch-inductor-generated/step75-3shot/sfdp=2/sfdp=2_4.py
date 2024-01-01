
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(16, 8)
 
    def forward(self, x1, x2):
        q = x1.softmax(dim=-1)
        k = x2.softmax(dim=-1)
        v = torch.randn(3, 8, 4)
        qk = torch.matmul(q, k.transpose(-2, -1)) 
        scaled_qk = qk.div(0.10000000149011612) 
        softmax_qk = scaled_qk.softmax(dim=-1) 
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=0.20000000298023224)
        output = dropout_qk.matmul(v)
        return output

# Initializing the model
model = Model()

# Inputs to the model
x1 = torch.randn(3, 4, 16)
x2 = torch.randn(3, 2, 16)
