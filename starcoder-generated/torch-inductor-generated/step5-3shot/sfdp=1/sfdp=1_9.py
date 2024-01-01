
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, t5, t6, t7):
        qk = torch.matmul(t5, t6.transpose(-2, -1))
        scaled_qk = qk.div(2.**(-1.))
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=0.3)
        output = dropout_qk.matmul(t7)
        return output

# Initializing the model
m = Model()

# Inputs to the model
t5 = torch.randn(4, 20, 40, 50)
t6 = torch.randn(4, 20, 30, 25)
t7 = torch.randn(4, 20, 30, 25)
