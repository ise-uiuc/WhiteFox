
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, Input0, Input1, in4, in5):
        in2 = torch.unsqueeze(torch.reshape(Input0, (-1, 56)), 0)
        in3 = torch.div(in2, in4)
        in7 = torch.reshape(in3, shape=(1, 1, 56))
        in8 = torch.add(in7, in5)
        in9 = torch.sigmoid(in8)
        in10 = torch.reshape(in9, shape=(1, 56, 56, 1))
        return
# Inputs to the model
Input0 = torch.randn(64, 56, 56)
Input1 = torch.randn(1, 56)
in4 = 0.1
in5 = 0.9
