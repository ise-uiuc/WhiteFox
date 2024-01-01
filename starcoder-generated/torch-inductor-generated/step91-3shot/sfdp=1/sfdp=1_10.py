
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.qk = torch.nn.Linear(30, 40)
        self.value = torch.nn.Linear(40, 20)
 
    def forward(self, input__1):
        qk = self.qk(input__1)
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(self.value)
        return output

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 768)
