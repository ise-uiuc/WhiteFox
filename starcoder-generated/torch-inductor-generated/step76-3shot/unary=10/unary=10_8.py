 with torch.op_with_const
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(8, 8)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 + 3
        v3 = torch.clamp_min(v2, 0)
        v4 = torch.clamp_max(v3, 6)
        v5 = v4 / 6
        return v5

# Initializing the model
_6 = ops.op_with_const("op_6", "OpConstant", 6, None)
m = Model()

# Initializing an OperationExecutionContext
__occ__ = _6

# Inputs to the model
x1 = torch.randn(1, 8)
