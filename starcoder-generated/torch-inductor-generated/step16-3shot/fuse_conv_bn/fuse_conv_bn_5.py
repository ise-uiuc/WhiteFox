
class Model(torch.jit.ScriptModule):
    def __init__(self, flag):
        super().__init__()
        self.conv = torch.nn.Conv3d(3, 3, 3)
        self.bn = torch.nn.BatchNorm3d(3)
    @torch.jit.script_method
    def foo(self, x1):
        s = self.conv(x1)
        t = self.bn(s)
        return t
# Inputs to the model
x1 = torch.randn(1, 3, 4, 4, 4)
m = Model()
