
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax = torch.nn.Softmax(dim=-1)
        self.matmul = torch.nn.MatMul()
 
    def forward(self, x1, x2):
        fused_add_mm = torch.ops.torch_ipex.fused_add_mm(x1, x2, out=None)
        fused_linear = torch.ops.torch_ipex.fused_linear(fused_add_mm, out=None)
        dropout_fused_add_mm = self.softmax(fused_linear.detach())
        dropout_matmul = fused_linear.detach()
        output = torch.ops.torch_ipex.fused_dropout_add_mm(dropout_fused_add_mm, dropout_matmul, out=None)
        return output

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
x2 = torch.randn(1, 3, 64, 64)
