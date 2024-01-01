
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(6, 3, 3, stride=2, padding=1, bias=False, dilation=1, groups=1, output_padding=1)
        self.batch_norm = torch.nn.BatchNorm2d(3)
    def forward(self, x4):
        t1 = self.conv_t(x4)
        t2 = t1 > 0
        t3 = t1 * -0.7
        t4 = torch.where(t2, t1, t3)
        t5 = t4 < -0.5
        t6 = torch.where(t5, t4, torch.full_like(t4, -0.5))
        return self.batch_norm(t6)
# Inputs to the model
x4 = torch.tensor([[[[1.4339, 1.0922, -0.6388, -0.1103, 0.1963],
                      [0.3688, -1.3627, 0.4187, -3.6826, 0.6972],
                      [0.3013, 0.6479, -0.8276, -0.6090, -0.3269],
                      [-1.0935, -0.6960, 0.7967, -0.1933, -1.1418]],
                     [[0.2520, 1.0449, 1.4477, 0.6704, 0.3320],
                      [-1.1170, -0.0059, 0.9746, 1.4818, 1.4852],
                      [0.3650, -0.6202, 0.9495, 0.5305, 0.0139],
                      [-0.0812, 2.4373, -0.4203, -0.2284, 1.8471]],
                     [[-0.9821, 1.2651, -0.0592, -1.8132, -0.7374],
                      [-0.0860, 0.6335, -0.3244, -0.3862, 0.1286],
                      [-0.3208, -0.0389, 0.5319, 0.5077, 1.1110],
                      [-0.7375, 2.5910, -1.9562, -0.2144, 1.4472]]]], dtype=torch.float32)
