
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d_5 = torch.nn.Conv2d(16, 4, kernel_size=(2, 2), stride=(2, 2))
    def forward(self, x1):
        v0 = 200000.0 * torch.tanh(torch.mean((-23.0*((0.999999761581421)*(((x1) * ((x1) * ((x1) * ((x1) * ((x1) * ((x1) * ((x1) * ((x1) * ((x1) * ((x1) * ((x1) * ((x1) * ((x1) * ((x1) * ((x1) * ((x1) * ((x1) * ((x1) * ((x1) * ((x1) * ((x1) * ((x1) * ((x1) * ((x1) * (5.6839699e-07*(torch.pow(x1, 4)))+(-11.195321*(torch.pow(x1, 3))))+(-1.9261676*(torch.pow(x1, 2))))+(-3.2055359e-05*(x1))))+(-0.019092241))+(-0.014644863))))))))))))))))))*(x1)), dim=3))
        v1 = self.conv2d_5(x1)
        v2 = torch.tanh(v1)
        v3 = torch.sigmoid(v2)
        v4 = v0 * v3
        return v4
# Inputs to the model
x1 = torch.randn(19, 4, 128, 128)
torch.manual_seed(0);
