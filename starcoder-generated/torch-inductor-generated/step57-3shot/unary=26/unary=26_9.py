
import torchvision as torchv
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torchv.models.densenet201()
    def forward(self, x11):
        x2 = x11.to(torch.float32)
        y1 = self.conv_t.features.conv0(x2)
        y2 = self.conv_t.features.norm0(y1)
        y3 = self.conv_t.features.relu0(y2)
        y4 = self.conv_t.features.pool0(y3)
        y5 = self.conv_t.features.denseblock1.denselayer1.conv1(y4)
        y6 = self.conv_t.features.denseblock1.denselayer1.norm1(y5)
        y7 = self.conv_t.features.denseblock1.denselayer1.relu1(y6)
        y8 = self.conv_t.features.denseblock1.denselayer1.conv2(y7)
        y9 = self.conv_t.features.denseblock1.denselayer1.norm2(y8)
        y10 = self.conv_t.features.denseblock1.denselayer1.relu2(y9)
        y11 = self.conv_t.features.denseblock1.denselayer1.conv3(y10)
        y12 = self.conv_t.features.denseblock1.denselayer1.norm3(y11)
        y13 = self.conv_t.features.denseblock1.denselayer1.relu3(y12)
        y14 = self.conv_t.features.denseblock1.denselayer1.conv4(y13)
        y15 = self.conv_t.features.denseblock1.denselayer1.norm4(y14)
        y16 = self.conv_t.features.denseblock1.denselayer1.relu4(y15)
        y18 = torch.max(y16, [2])[0]
        y17 = torch.squeeze(y18, [2, 3])
        y19 = self.conv_t.features.denseblock1.denselayer2.conv1(y17)
        y20 = self.conv_t.features.denseblock1.denselayer2.norm1(y19)
        y21 = self.conv_t.features.denseblock1.denselayer2.relu1(y20)
        y22 = self.conv_t.features.denseblock1.denselayer2.conv2(y21)
        y23 = self.conv_t.features.denseblock1.denselayer2.norm2(y22)
        y24 = self.conv_t.features.denseblock1.denselayer2.relu2(y23)
        y25 = self.conv_t.features.denseblock1.denselayer2.conv3(y24)
        y26 = self.conv_t.features.denseblock1.denselayer2.norm3(y25)
        y27 = self.conv_t.features.denseblock1.denselayer2.relu3(y26)
        y28 = self.conv_t.features.denseblock1.denselayer2.conv4(y27)
        y29 = self.conv_t.features.denseblock1.denselayer2.norm4(y28)
        y30 = self.conv_t.features.denseblock1.denselayer2.relu4(y29)
        y31 = self.conv_t.features.denseblock1.denselayer3.conv1(y30)
        y32 = self.conv_t.features.denseblock1.denselayer3.norm1(y31)
        y33 = self.conv_t.features.denseblock1.denselayer3.relu1(y32)
        y34 = self.conv_t.features.denseblock1.denselayer3.conv2(y33)
        y35 = self.conv_t.features.denseblock1.denselayer3.norm2(y34)
        y36 = self.conv_t.features.denseblock1.denselayer3.relu2(y35)
        y37 = self.conv_t.features.denseblock1.denselayer3.conv3(y36)
        y38 = self.conv_t.features.denseblock1.denselayer3.norm3(y37)
        y39 = self.conv_t.features.denseblock1.denselayer3.relu3(y38)
        y40 = self.conv_t.features.denseblock1.denselayer3.conv4(y39)
        y41 = self.conv_t.features.denseblock1.denselayer3.norm4(y40)
        y42 = self.conv_t.features.denseblock1.denselayer3.relu4(y41)
        y44 = torch.add(y42, self.conv_t.features.denseblock1.denselayer4.conv1(y42))
        y43 = self.conv_t.features.denseblock1.denselayer4.norm1(y44)
        y45 = self.conv_t.features.denseblock1.denselayer4.relu1(y43)
        y46 = self.conv_t.features.denseblock1.denselayer4.conv2(y45)
        y47 = self.conv_t.features.denseblock1.denselayer4.norm2(y46)
        y48 = self.conv_t.features.denseblock1.denselayer4.relu2(y47)
        y49 = self.conv_t.features.denseblock1.denselayer4.conv3(y48)
        y50 = self.conv_t.features.denseblock1.denselayer4.norm3(y49)
        y51 = self.conv_t.features.denseblock1.denselayer4.relu3(y50)
        y52 = self.conv_t.features.denseblock1.denselayer4.conv4(y51)
        y53 = self.conv_t.features.denseblock1.denselayer4.norm4(y52)
        y54 = self.conv_t.features.denseblock1.denselayer4.relu4(y53)
        y55 = self.conv_t.features.denseblock1.denselayer5.conv1(y54)
        y56 = self.conv_t.features.denseblock1.denselayer5.norm1(y55)
        y57 = self.conv_t.features.denseblock1.denselayer5.relu1(y56)
        y58 = self.conv_t.features.denseblock1.denselayer5.conv2(y57)
        y59 = self.conv_t.features.denseblock1.denselayer5.norm2(y58)
        y60 = self.conv_t.features.denseblock1.denselayer5.relu2(y59)
        y61 = self.conv_t.features.denseblock1.denselayer5.conv3(y60)
        y62 = self.conv_t.features.denseblock1.denselayer5.norm3(y61)
        y63 = self.conv_t.features.denseblock1.denselayer5.relu3(y62)
        y64 = self.conv_t.features.denseblock1.denselayer5.conv4(y63)
        y65 = self.conv_t.features.denseblock1.denselayer5.norm4(y64)
        y66 = self.conv_t.features.denseblock1.denselayer5.relu4(y65)
        y68 = torch.max(y66, [2])[0]
        y67 = torch.squeeze(y68, [2, 3])
        y69 = self.conv_t.features.denseblock1.denselayer6.conv1(y67)
        y70 = self.conv_t.features.denseblock1.denselayer6.norm1(y69)
        y71 = self.conv_t.features.denseblock1.denselayer6.relu1(y70)
        y72 = self.conv_t.features.denseblock1.denselayer6.conv2(y71)
        y73 = self.conv_t.features.denseblock1.denselayer6.norm2(y72)
        y74 = self.conv_t.features.denseblock1.denselayer6.relu2(y73)
        y75 = self.conv_t.features.denseblock1.denselayer6.conv3(y74)
        y76 = self.conv_t.features.denseblock1.denselayer6.norm3(y75)
        y77 = self.conv_t.features.denseblock1.denselayer6.relu3(y76)
        y78 = self.conv_t.features.denseblock1.denselayer6.conv4(y77)
        y79 = self.conv_t.features.denseblock1.denselayer6.norm4(y78)
        y80 = self.conv_t.features.denseblock1.denselayer6.relu4(y79)
        y81 = self.conv_t.features.denseblock1.denselayer7.conv1(y80)
        y82 = self.conv_t.features.denseblock1.denselayer7.norm1(y81)
        y83 = self.conv_t.features.denseblock1.denselayer7.relu1(y82)
        y84 = self.conv_t.features.denseblock1.denselayer7.conv2(y83)
        y85 = self.conv_t.features.denseblock1.denselayer7.norm2(y84)
        y86 = self.conv_t.features.denseblock1.denselayer7.relu2(y85)
        y87 = self.conv_t.features.denseblock1.denselayer7.conv3(y86)
        y88 = self.conv_t.features.denseblock1.denselayer7.norm3(y87)
        y89 = self.conv_t.features.denseblock1.denselayer7.relu3(y88)
        y90 = self.conv_t.features.denseblock1.denselayer7.conv4(y89)
        y91 = self.conv_t.features.denseblock1.denselayer7.norm4(y90)
        y92 = self.conv_t.features.denseblock1.denselayer7.relu4(y91)
        y94 = torch.add(y92, self.conv_t.features.denseblock1.denselayer8.conv1(y92))
        y93 = self.conv_t.features.denseblock1.denselayer8.norm1(y94)
        y95 = self.conv_t.features.denseblock1.denselayer8.relu1(y93)
        y96 = self.conv_t.features.denseblock1.denselayer8.conv2(y95)
        y97 = self.conv_t.features.denseblock1.denselayer8.norm2(y96)
        y98 = self.conv_t.features.denseblock1.denselayer8.relu2(y97)
        y99 = self.conv_t.features.denseblock1.denselayer8.conv3(y98)
        y100 = self.conv_t.features.denseblock1.denselayer8.norm3(y99)
        y101 = self.conv_t.features.denseblock1.denselayer8.relu3(y100)
        y102 = self.conv_t.features.denseblock1.denselayer8.conv4(y101)
        y103 = self.conv_t.features.denseblock1.denselayer8.norm4(y102)
        y104 = self.conv_t.features.denseblock1.denselayer8.relu4(y103)
        y105 = self.conv_t.features.denseblock1.denselayer9.conv1(y104)
        y106 = self.conv_t.features.denseblock1.denselayer9.norm1(y105)
        y107 = self.conv_t.features.denseblock1.denselayer9.relu1(y106)
        y108 = self.conv_t.features.denseblock1.denselayer9.conv2(y107)
        y109 = self.conv_t.features.denseblock1.denselayer9.norm2(y108)
        y110 = self.conv_t.features.denseblock1.denselayer9.relu2(y109)
        y111 = self.conv_t.features.denseblock1.denselayer9.conv3(y110)
        y112 = self.conv_t.features.denseblock1.denselayer9.norm3(y111)
        y113 = self.conv_t.features.denseblock1.denselayer9.relu3(y112)
        y114 = self.conv_t.features.denseblock1.denselayer9.conv4(y113)
        y115 = self.conv_t.features.denseblock1.denselayer9.norm4(y114)
        y116 = self.conv_t.features.denseblock1.denselayer9.relu4(y115)
        y118 = torch.max(y116, [2])[0]
        y117 = torch.squeeze(y118, [2, 3])
        y119 = self.conv_t.features.denseblock1.denselayer10.conv1(y117)
        y120 = self.conv_t.features.denseblock1.denselayer10.norm1(y119)
        y121 = self.conv_t.features.denseblock1.denselayer10.relu1(y120)
        y122 = self.conv_t.features.denseblock1.denselayer10.conv2(y121)
        y123 = self.conv_t.features.denseblock1.denselayer10.norm2(y122)
        y124 = self.conv_t.features.denseblock1.denselayer10.relu2(y123)
        y125 = self.conv_t.features.denseblock1.denselayer10.conv3(y124)
        y126 = self.conv_t.features.denseblock1.denselayer10.norm3(y125)
        y127 = self.conv_t.features.denseblock1.denselayer10.relu3(y126)
        y128 = self.conv_t.features.denseblock1.denselayer10.conv4(y127)
        y129 = self.conv_t.features.denseblock1.denselayer10.norm4(y128)
        y130 = self.conv_t.features.denseblock1.denselayer10.relu4(y129)
        y131 = self.conv_t.features.denseblock1.denselayer11.conv1(y130)
        y132 = self.conv_t.features.denseblock1.denselayer11.norm1(y131)
        y133 = self.conv_t.features.denseblock1.denselayer11.relu1(y132)
        y134 = self.conv_t.features.denseblock1.denselayer11.conv2(y133)
        y135 = self.conv_t.features.denseblock1.denselayer11.norm2(y134)
        y136 = self.conv_t.features.denseblock1.denselayer11.relu2(y135)
        y137 = self.conv_t.features.denseblock1.denselayer11.conv3(y136)
        y138 = self.conv_t.features.denseblock1.denselayer11.norm3(y137)
        y139 = self.conv_t.features.denseblock1.denselayer11.relu3(y138)
        y140 = self.conv_t.features.denseblock1.denselayer11.conv4(y139)
        y141 = self.conv_t.features.denseblock1.denselayer11.norm4(y140)
        y142 = self.conv_t.features.denseblock1.denselayer11.relu4(y1