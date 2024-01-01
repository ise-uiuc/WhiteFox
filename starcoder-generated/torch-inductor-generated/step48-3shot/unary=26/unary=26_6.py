
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv_t_1 = torch.nn.ConvTranspose2d(200,509, kernel_size=[1,4], stride=[2,1], bias=False)
        self.conv_t_2 = torch.nn.ConvTranspose2d(509,357, kernel_size=[5,2], stride=[2,1], bias=False)
        self.conv_t_3 = torch.nn.ConvTranspose2d(357,200, kernel_size=[3,2], stride=[1,1], bias=False)
    def forward(self, input1, input2):
        v1 = self.conv_t_1(input1)
        v2 = self.conv_t_2(input2)
        v3 = self.conv_t_3(v2 + v1)
        t2 = v3 > 0
        t3 = v3 * -1.65
        v4 = torch.where(t2, v3, t3)
        t4 = v4 > 0
        t5 = v4 * 0.5
        v5 = torch.where(t4, v4, t5)
        return v5
# Inputs to the model
input1 = torch.randn([868,565,1,1])
input2 = torch.randn([868,116,-15,34])
