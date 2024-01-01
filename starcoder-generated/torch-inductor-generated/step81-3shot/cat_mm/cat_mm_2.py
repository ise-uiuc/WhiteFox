
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, inputs1, inputs2, inputs3):
        t0 = torch.cat([
            torch.mm(inputs1, inputs2),
            torch.mm(inputs1, inputs2),
            torch.mm(inputs1, inputs2),
        ], dim=1)
        t1 = torch.cat([
            torch.mm(inputs1, inputs3),
            torch.mm(inputs1, inputs3),
            torch.mm(inputs1, inputs3),
        ], dim=1)
        t2 = torch.cat([
            torch.mm(inputs1, inputs3),
            torch.mm(inputs1, inputs3),
            torch.mm(inputs1, inputs3),
        ], dim=1)
        return torch.cat([t0, t1, t2], dim=1)
# Inputs to the model
inputs1 = torch.randn(3, 3)
inputs2 = torch.randn(3, 3)
inputs3 = torch.randn(3, 3)
