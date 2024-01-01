
class Model(torch.nn.Module):
    def forward(self, in1, in2, in3, in4):
        t1 = torch.nn.functional.matmul(in1, in2)
        t2 = torch.nn.functional.matmul(in3, in4)
        return t1 * t2
# Inputs to the model
in1 = torch.randn(16,8)
in2 = torch.randn(8,16)
in3 = torch.randn(16,8)
in4 = torch.randn(8,16)
