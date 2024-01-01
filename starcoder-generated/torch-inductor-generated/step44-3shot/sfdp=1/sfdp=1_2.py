
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(64, 256)
 
    def forward(self, x1, x2):
        v1 = self.linear(x1)
        qk = torch.matmul(v1, x2.transpose(-2, -1))
        inv_scale_factor = 1 / numpy.sqrt(v1.shape[-1])
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=0.1)
        return dropout_qk

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 256)
x2 = torch.randn(1, 512, 256)
