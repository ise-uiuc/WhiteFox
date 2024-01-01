
class model_0(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.deconv0 = torch_op.Deconv((1,16,120,120))
        self.relu0 = torch.nn.ReLU()
    def forward(self, input_1):
        var1 = self.deconv0(input_1)
        var2 = self.relu0(var1)
        return var2
# Inputs to the model
input_1 = torch.randn(1, 16, 120, 120)
