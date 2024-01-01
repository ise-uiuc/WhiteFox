
class Model(torch.nn.Module):
    def forward(self, input1):
        t1 = torch.mm(input1, input1)
        t2 = t1[..., 0]
        t3 = torch.mm(input1, input1)
        t4 = t3[..., 100]
        t5 = torch.mm(input1, input1)
        t6 = t5[..., 5000] * t5[..., 10000]
        return t2 + t4 + t5 + t6
# Inputs to the model
input1 = torch.randn(512, 512)
