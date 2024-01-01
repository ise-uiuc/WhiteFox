


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t1 = torch.nn.ConvTranspose2d(3, 6, 6, stride=1, padding=0, bias=False)
        self.conv_t2 = torch.nn.ConvTranspose2d(6, 3, 2, stride=2, output_padding=1, bias=False)
        self.conv_t3 = torch.nn.ConvTranspose2d(3, 6, 2, stride=2, padding=1, output_padding=1, bias=False)
    def forward(self, x3):
        x1 = self.conv_t1(x3)
        x2 = self.conv_t2(x1)
        x3 = self.conv_t3(x2)
        return x3
# Inputs to the model
x3 = torch.tensor([[[
    [0.3511506623744965, 0.796820960521698],
    [0.12747512657642365, 0.182371666674614],
    [0.8047687888145447, 0.566534462928772]
], [
    [0.7960432958602905, 0.15151918814659119],
    [0.02814149165916443, 0.4471722857952118],
    [0.9902585501670837, 0.7442953705787659]
]]])
