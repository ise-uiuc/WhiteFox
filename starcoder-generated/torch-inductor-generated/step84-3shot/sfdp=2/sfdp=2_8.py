
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, __input__, __input__1, __input__2):
        qk = torch.matmul(__input__, __input__1.transpose(-2, -1))
        scaled_qk = qk.div(1e-7) 
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=0.5)
        output = dropout_qk.matmul(__input__2)
        return output

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 768, 64)
x2 = torch.randn(1, 64, 768)
v1 = torch.randn(1, 64, 768)
