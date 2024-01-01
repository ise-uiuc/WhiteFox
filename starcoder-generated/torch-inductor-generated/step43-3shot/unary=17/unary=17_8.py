
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.ConvTranspose3d(1, 64, 1, stride=(1, 2, 2))
        self.conv2 = torch.nn.Conv2d(64, 128, kernel_size=(3, 3), stride=1, padding=1)
        self.conv3 = torch.nn.ConvTranspose2d(128, 64, kernel_size=(2, 2), stride=(1, 2), padding=1)
    def forward(self, x):
        s0 = x.shape
        t0 = x.view(-1, 1, s0[2], s0[3], s0[4])
        t1 = self.conv1(t0)
        t1 = torch.relu(t1)
        t2 = self.conv2(t1)
        t2 = torch.tanh(t2)
        t3 = self.conv3(t2)
        s1 = t3.size()
        t4 = t3.view(-1,
                     s1[1] * s1[2] * s1[3],
                     s1[4])
        t5 = torch.relu(t4)
        t6 = t5.unsqueeze(2)
        return t6
# Inputs to the model
x = torch.randn(1, 1, 6, 32, 32)
