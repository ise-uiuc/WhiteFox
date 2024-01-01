
class FwdBlock(torch.nn.Module):
    def __init__(self, inp, out):
        super(FwdBlock, self).__init__()
        self.ops = torch.nn.Sequential(torch.nn.Conv1d(inp, out, 1, 1, 0), torch.nn.ReLU(),torch.nn.Conv1d(out, out, 1, 1, 0), torch.nn.ReLU(),torch.nn.Conv1d(out, out, 1, 1, 0), torch.nn.ReLU())
    def forward(self, x):
        return self.ops(x)
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # Block1
        self.features = FwdBlock(8, 16)
        # self.layer3 = nn.Sequential(FwdBlock(16, 16), FwdBlock(16, 16))
        # Block2
        self.extra = FwdBlock(16, 16)
    def forward(self, x):
        split_tensors = torch.split(x, [1, 1, 1, 1, 1, 1, 1, 1], dim=1)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        return (concatenated_tensor, torch.split(x, [1, 1, 1, 1, 1, 1, 1, 1], dim=1))
# Inputs to the model
x1 = torch.randn(1, 8, 240)
