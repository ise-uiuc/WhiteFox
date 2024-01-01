
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scaled_qk = torch.nn.Softmax(dim=-1)
 
    def forward(self, x1, x2):
        qk = torch.matmul(x1, x2.transpose(-2, -1))
        scaled_qk = qk.mul(1e8)
        softmax_qk = self.scaled_qk(scaled_qk)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=0.5)
        output = dropout_qk.matmul(x2)
        return output

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 32, 512, 64)
x2 = torch.randn(1, 32, 512, 64)
