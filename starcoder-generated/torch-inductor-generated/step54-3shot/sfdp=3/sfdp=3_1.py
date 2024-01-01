
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale_factor = 1
        self.dropout_p = 0.2
 
    def forward(self, x1, x2):
        qk = torch.matmul(x1, x2.transpose(-2, -1))
        scaled_qk = qk.mul(self.scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p)
        output = dropout_qk.matmul(value)
        return output
 
# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 8, 32, 48)
x2 = torch.randn(1, 8, 48, 32)
