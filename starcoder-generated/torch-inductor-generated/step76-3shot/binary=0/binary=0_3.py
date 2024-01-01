
def conv(input, stride1):
    conv = torch.nn.Conv2d(3, 34, 3, stride=(stride1), padding=2)
    output = conv(input)
    return output
# Inputs to the model
input = torch.randn(2, 3, 4, 4)
stride1 = 1
