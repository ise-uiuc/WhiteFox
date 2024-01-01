
class Model(torch.nn.Module):
    def forward(self, in1, bn1, in2, bn2, in3, bn3, in4, bn4):
        t1 = torch.cat((bn1, bn2), dim=1)
        t = torch.mm(t1, t1.transpose(0,1)) + bn3
        return t.mm(bn4)
# Inputs to the model
in1 = torch.randn(4, 4, 4)
in2 = torch.randn(4, 4, 3)
in3 = torch.randn(3, 1)
in4 = torch.randn(4, 4)
