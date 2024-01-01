
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.randn(1, 1, 1))
    def forward(self, input_tensor):
        t1 = torch.cat([input_tensor, input_tensor], dim=1)
        t2 = t1.permute(0, 2, 1)
        t3 = t2.sigmoid()
        return (input_tensor + torch.mul(self.bias, t3))
# Inputs to the model
x = torch.randn(2, 4, 4)
