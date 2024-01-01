
class Block(torch.nn.Module):
    def __init__(self, inp, out):
        super().__init__()
        self.op1 = torch.nn.BatchNorm2d(inp, affine=False, track_running_stats=False)
        self.op2 = torch.nn.ReLU(inplace=True)
        self.op3 = torch.nn.Linear(out, out, out)
    def forward(self, v1):
        op4 = self.op3(self.op2(self.op1(v1)))
        concat_tensors = torch.cat([1, op4, 2, 3], dim=0)
        return (concat_tensors, torch.split([1, op4], [1, 1, 1], dim=0))
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([Block(4, 3), Block(3, 4)])
    def forward(self, v1):
        op4 = self.layers[0](v1)
        split_tensors = torch.split(v1, [1, 1], dim=2)
        concatenated_tensor = torch.cat(split_tensors, dim=2)
        op4 += concatenated_tensor
        op5 = torch.cat([op4, 1, 2, 3], dim=1)
        op6 = ((self.layers[1](op5)) + (op5))
        return (op6, torch.split((self.layers[1](op5)), [1, 1, 1], dim=1) + torch.split(op5, [1, 1, 1], dim=1))
# Inputs to the model
x1 = torch.randn(6, 4)
