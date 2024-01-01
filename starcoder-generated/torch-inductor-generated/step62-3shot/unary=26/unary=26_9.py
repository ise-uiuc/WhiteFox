
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t_1 = torch.nn.ConvTranspose2d(1,3,2)
        self.conv_t_2 = torch.nn.ConvTranspose2d(3,1,3,stride=3,padding=1)

    def forward(self, images):
        out = torch.zeros((images.shape[0],3,10,10), device='cpu')
        mask = torch.zeros((images.shape[0],1,10), device='cpu').type(torch.LongTensor)

        conv_1 = self.conv_t_1(images)
        conv_15 = F.max_pool2d(conv_1,2,2)
        conv_2 = self.conv_t_2(conv_15)
        out = torch.where((conv_2>conv_15).repeat(1,3,1,1),conv_2,conv_15)
        mask = torch.where((conv_2>conv_15).repeat(1,1,1),torch.ones(conv_2.shape),torch.zeros(conv_2.shape)).type(torch.LongTensor)
        return out * mask.repeat(1,3,1,1), mask
# Inputs to the model
images = torch.randn(3,1,17,19)
