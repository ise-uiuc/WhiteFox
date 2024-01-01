
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2):
        qk = torch.matmul(x1, x2.transpose(-2, -1))
        inv_scale_factor = math.sqrt(x2.shape[-1])
        dropout_p = 0.0
        softmax_qk = torch.nn.functional.dropout(qk.softmax(dim=-1), p=dropout_p)
        output = softmax_qk.matmul(x2)
        return output

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(10, 4, 51)
x2 = torch.randn(10, 51, 64)
