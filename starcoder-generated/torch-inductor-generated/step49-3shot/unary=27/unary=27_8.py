
class Model(torch.nn.Module):
    def __init__(self, min, max):
        super().__init__()
        self.mul1 = torch.nn.quantized.FloatFunctional()
        self.mul1.register_non_leaf_module("mul1", lambda _: torch.quantize_per_tensor(torch.tensor(5.5), 0.35314310, 0, torch.qint8))
        self.clamp_min1 = torch.nn.quantized.FloatFunctional()
        self.clamp_min1.register_non_leaf_module("clamp_min1", lambda ctx: torch._VF._existing_ OpModule(op="clamp_min", domain="quantized", inputs=(a, ctx[1]), outputs=(a,), args=(ctx[0],), kwargs={}))
        self.add1 = torch.nn.quantized.FloatFunctional()
        self.add1.register_non_leaf_module("add1", lambda ctx: torch._VF._existing_ OpModule(op="add", domain="quantized", inputs=(ctx[2], ctx[3]), outputs=(a,), args=(), kwargs={}))
    def forward(self, x1):
        a = self.mul1(self.clamp_min1(x1, min, max), min, max)
        a = self.clamp_min1(a, min, max)
        a = self.add1(a, 5)
        return torch.nn.quantized.FloatFunctional.cat([x1, a], dim=1)
min = 10
max = 4
# Inputs to the model
x1 = torch.randn(1, 15, 15, 15)
