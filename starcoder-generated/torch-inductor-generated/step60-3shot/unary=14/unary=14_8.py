
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__();
        self.convtranspose = torch.nn.ConvTranspose2d(13, 4, 1, stride=1, padding=0);
    def forward(self, input1):
        output = self.convtranspose(input1);
        return output;
# Inputs to the model
input1 = torch.randn(1, 13, 128, 3)
