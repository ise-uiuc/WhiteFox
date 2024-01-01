
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale_factor = torch.nn.Parameter(torch.Tensor([0.5]))
        self.dropout_p = torch.nn.Parameter(torch.Tensor([0.7]))
        self.value = torch.nn.Parameter(torch.Tensor([[1.0, 2.0]]))
 
    def forward(self, x1, x2):
        qk = torch.matmul(x1, x2.transpose(-2, -1))
        scaled_qk = qk.mul(self.scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p)
        output = dropout_qk.matmul(self.value)
        return output
 
# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 8, 64)
x2 = torch.randn(1, 8, 64)
