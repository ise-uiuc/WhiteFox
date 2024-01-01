
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout_p = 0.1
 
    def forward(self, __input0__, __input1__):
        qk = torch.matmul(__input0__, __input1__.transpose(-2, -1))
        scale_factor = (1.0 / math.sqrt(__input0__.shape[-1]))
        scaled_qk = qk.mul(scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p)
        output = dropout_qk.matmul(__input1__)
        return output

# Initializing the model
m = Model()

# Inputs to the model
__x0__ = torch.randn(1, 12, 512, 64)
__x1__ = torch.randn(1, 12, 64, 512)
