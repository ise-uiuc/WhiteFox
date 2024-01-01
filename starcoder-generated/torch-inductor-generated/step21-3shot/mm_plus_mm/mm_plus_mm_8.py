
class Model(nn.Module):
    def forward(self, input1, input2, input3, input4):
            t1 = nn.Linear(512, 128)(input3)
            t2 = nn.Linear(512, 128)(input1)
            t3 = t1 + t2
            t4 = torch.mm(t3, nn.Linear(128,512)(input4))
# Inputs to the model
input1 = torch.randn(8, 512)
input2 = torch.randn(8, 512)
input3 = torch.randn(8, 512)
input4 = torch.randn(512, 512)
