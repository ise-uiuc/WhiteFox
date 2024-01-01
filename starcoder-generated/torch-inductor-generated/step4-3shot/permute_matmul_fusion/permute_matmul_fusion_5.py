
class Model(torch.jit.ScriptModule):
    @torch.jit.script_method
    def forward(self, x1, x2):
        v1 = x1.permute(0, 2, 1)
        v2 = torch.matmul(v1, x2)
        return v2
# Inputs to the model
x1 = torch.randn(1, 2, 2)
x2 = torch.randn(1, 2, 2)
