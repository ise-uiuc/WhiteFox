
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 8)
 
    def forward(self, x, y):
        z = torch.cat((x, y), dim=-1)
        q = self.linear(z)
        k = self.linear(z)
        v = self.linear(z)
        qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = qk.div(8)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=0.39999995)
        output = dropout_qk.matmul(v)
        return output

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(2, 3)
y = torch.randn(2, 3)
