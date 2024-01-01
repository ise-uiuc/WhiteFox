
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(64, 64, 0, stride=2, bias=False)
    def forward(self, input1):
        t8 = self.conv_t(input1)
        t9 = torch.max(t8, -3.20109)
        t10 = torch.nn.functional.relu(t9.neg())
        return t10
# Inputs to the model
input1 = torch.randn(1, 64, 7, 14)
