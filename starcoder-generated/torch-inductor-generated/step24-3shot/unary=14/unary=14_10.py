
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.convTranspose_15 = torch.nn.ConvTranspose2d(20, 60, 3, stride=2, padding=1)
        self.convTranspose_25 = torch.nn.ConvTranspose2d(60, 68, 3, stride=2, padding=1)
        self.convTranspose_35 = torch.nn.ConvTranspose2d(68, 60, 3, stride=2, padding=1)
        self.convTranspose_45 = torch.nn.ConvTranspose2d(60, 7, 3, stride=2, padding=1)
    def forward(self, x):
        output = self.convTranspose_15(x)
        output = output + x
        output = self.convTranspose_25(output)
        output = self.convTranspose_35(output)
        output = output + x
        output = self.convTranspose_45(output)
        return output
# Inputs to the model
x = torch.randn(1, 20, 30, 20)
