
class Model(jit.ScriptModule):
    @torch.jit.script_method
    def forward(self, input1, input2, input3, input4):
        return torch.mm(input1, input2) * torch.mm(input3, input4) // 2
# Inputs to the model
input1 = torch.randn(100, 100)
input2 = torch.randn(100, 100)
input3 = torch.randn(100, 100)
input4 = torch.randn(100, 100)
