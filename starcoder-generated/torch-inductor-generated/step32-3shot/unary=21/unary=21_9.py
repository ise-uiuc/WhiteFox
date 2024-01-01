
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # NCHW input, 1 output feature map, size 5x7 and a padding of 3
        # (1 input channel, 1 output channel, 5 kernel size, 7 kernel size)
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(5, 7), padding=(3, 3))
        # NCHW input, 1 output feature map, size 3x3 and a padding of 1
        # (1 input channel, 1 output channel, 3 kernel size, 3 kernel size)
        self.conv2 = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(3, 3), padding=1)
        # NCHW input, 64 output feature map, size 3x3 and a padding of 1
        # (1 input channel,64 output channel, 3 kernel size, 3 kernel size)
        self.conv3 = torch.nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(3, 3), padding=1)
        # NCHW input, 64 output feature map, size 3x3 and a padding of 1
        # (64 input channel,64 output channel, 3 kernel size, 3 kernel size)
        self.conv4 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1)
        # NCHW input, 64 output feature map, size 3x3 and a padding of 1
        # (64 input channel,64 output channel, 3 kernel size, 3 kernel size)
        self.conv5 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1)
        # NCHW input, 128 output feature map, size 3x3 and a padding of 0
        # (64 input channel,128 output channel, 3 kernel size, 3 kernel size)
        self.conv6 = torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=0)
        # NCHW input, 128 output feature map, size 3x3 and a padding of 1
        # (128 input channel,128 output channel, 3 kernel size, 3 kernel size)
        self.conv7 = torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1)
        # NCHW input, 256 output feature map, size 3x3 and a padding of 1
        # (128 input channel,256 output channel, 3 kernel size, 3 kernel size)
        self.conv8 = torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding=1)
        # NCHW input, 256 output feature map, size 3x3 and a padding of 1
        # (256 input channel,256 output channel, 3 kernel size, 3 kernel size)
        self.conv9 = torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), padding=1)
        # NCHW input, 256 output feature map, size 3x3 and a padding of 1
        # (256 input channel,256 output channel, 3 kernel size, 3 kernel size)
        self.conv10 = torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), padding=1)
        # NCHW input, 512 output feature map, size 3x3 and a padding of 0
        # (256 input channel,512 output channel, 3 kernel size, 3 kernel size)
        self.conv11 = torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), padding=0)
        # NCHW input, 512 output feature map, size 3x3 and a padding of 1
        # (512 input channel,512 output channel, 3 kernel size, 3 kernel size)
        self.conv12 = torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=1)
        # NCHW input, 256 output feature map, size 3x3 and a padding of 0
        # (512 input channel,256 output channel, 3 kernel size, 3 kernel size)
        self.conv13 = torch.nn.Conv2d(in_channels=512, out_channels=256, kernel_size=(3, 3), padding=0)
        # NCHW input, 256 output feature map, size 3x3 and a padding of 1
        # (256 input channel,256 output channel, 3 kernel size, 3 kernel size)
        self.conv14 = torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), padding=1)
        # NCHW input, 256 output feature map, size 3x3 and a padding of 1
        # (256 input channel,256 output channel, 3 kernel size, 3 kernel size)
        self.conv15 = torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), padding=1)
        # NCHW input, 512 output feature map, size 3x3 and a padding of 0
        # (256 input channel,512 output channel, 3 kernel size, 3 kernel size)
        self.conv16 = torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), padding=0)
        # NCHW input, 512 output feature map, size 3x3 and a padding of 1
        # (512 input channel,512 output channel, 3 kernel size, 3 kernel size)
        self.conv17 = torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=1)
        # NCHW input, 128 output feature map, size 3x3 and a padding of 0
        # (64 input channel,128 output channel, 3 kernel size, 3 kernel size)
        self.conv18 = torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=0)
        # NCHW input, 128 output feature map, size 3x3 and a padding of 1
        # (128 input channel,128 output channel, 3 kernel size, 3 kernel size)
        self.conv19 = torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1)
        #NCHW input, 128 output feature map, size 3x3 and a padding of 1
        #(128 input channel,128 output channel, 3 kernel size, 3 kernel size)
        self.conv20 = torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1)
        # NCHW input, 256 output feature map, size 3x3 and a padding of 1
        # (128 input channel,256 output channel, 3 kernel size, 3 kernel size)
        self.conv21 = torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding=1)
        # NCHW input, 256 output feature map, size 3x3 and a padding of 1
        # (256 input channel,256 output channel, 3 kernel size, 3 kernel size)
        self.conv22 = torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), padding=1)
        # NCHW input, 256 output feature map, size 3x3 and a padding of 1
        # (256 input channel,256 output channel, 3 kernel size, 3 kernel size)
        self.conv23 = torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), padding=1)
        # NCHW input, 512 output feature map, size 3x3 and a padding of 0
        # (256 input channel,512 output channel, 3 kernel size, 3 kernel size)
        self.conv24 = torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), padding=0)
        # NCHW input, 512 output feature map, size 3x3 and a padding of 1
        # (512 input channel,512 output channel, 3 kernel size, 3 kernel size)
        self.conv25 = torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=1)
        # NCHW input, 128 output feature map, size 3x3 and a padding of 0
        # (512 input channel,128 output channel, 3 kernel size, 3 kernel size)
        self.conv26 = torch.nn.Conv2d(in_channels=512, out_channels=128, kernel_size=(3, 3), padding=0)
        # NCHW input, 128 output feature map, size 3x3 and a padding of 1
        # (128 input channel,128 output channel, 3 kernel size, 3 kernel size)
        self.conv27 = torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1)
        # NCHW input, 256 output feature map, size 3x3 and a padding of 1
        # (128 input channel,256 output channel, 3 kernel size, 3 kernel size)
        self.conv28 = torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding=1)
        # NCHW input, 256 output feature map, size 3x3 and a padding of 1
        # (256 input channel,256 output channel, 3 kernel size, 3 kernel size)
        self.conv29 = torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), padding=1)
        # NCHW input, 256 output feature map, size 3x3 and a padding of 1
        # (256 input channel,256 output channel, 3 kernel size, 3 kernel size)
        self.conv30 = torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), padding=1)
        # NCHW input, 256 output feature map, size 3x3 and a padding of 1
        # (256 input channel,256 output channel, 3 kernel size, 3 kernel size)
        self.conv31 = torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), padding=1)
        # NCHW input, 512 output feature map, size 3x3 and a padding of 0
        # (256 input channel,512 output channel, 3 kernel size, 3 kernel size)
        self.conv32 = torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), padding=0)
        # NCHW input, 512 output feature map, size 3x3 and a padding of 1
        # (512 input channel,512 output channel, 3 kernel size, 3 kernel size)
        self.conv33 = torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=1)
        # NCHW input, 128 output feature map, size 3x3 and a padding of 0
        # (512 input channel,128 output channel, 3 kernel size, 3 kernel size)
        self.conv34 = torch.nn.Conv2d(in_channels=512, out_channels=128, kernel_size=(3, 3), padding=0)
        # NCHW input, 128 output feature map, size 3x3 and a padding of 1
        # (128 input channel,128 output channel, 3 kernel size, 3 kernel size)
        self.conv35 = torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1)
        # NCHW input, 128 output feature map, size 3x3 and a padding of 1
        # (128 input channel,128 output channel, 3 kernel size, 3 kernel size)
        self.conv36 = torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1)
        # NCHW input, 128 output feature map, size 3x3 and a padding of 1
        # (128 input channel,128 output channel, 3 kernel size, 3 kernel size)
        self.conv37 = torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1)
        # NCHW input, 128 output feature map, size 3x3 and a padding of 1
        # (128 input channel,128 output channel, 3 kernel size, 3 kernel size)
        self.conv38 = torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1)
        # NCHW input, 256 output feature map, size 3x3 and a padding of 1
        # (128 input channel,256 output channel, 3 kernel size, 3 kernel size)
        self.conv39 = torch.nn.Conv2d(in