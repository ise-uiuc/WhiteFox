
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.other_conv = torch.nn.Conv2d(8, 8, 9)
    def forward(self, input_tensor):
        t1 = self.conv(input_tensor)
        t2 = 3 + t1 # Add 3 to the output of the convolution
        t3 = t2.clamp_(min=0, max=6) # Clamp the output of the addition operation to a minimum of 0 and a maximum of 6
        t4 = t3.div(6) # Divide the output of the previous operation by 6
        t5 = t4.permute(0, 2, 3, 1)
        t6 = self.other_conv(t5)
        t7 = t6.clamp_max(6)
        t8 = t7.div(6)
        t9 = t8.permute(0, 3, 1, 2)
        t10 = t9.clamp_max(6)
        t11 = t10.div(6)
        return t11
# Inputs to the model
input_tensor = torch.randn(3, 3, 64, 64)
