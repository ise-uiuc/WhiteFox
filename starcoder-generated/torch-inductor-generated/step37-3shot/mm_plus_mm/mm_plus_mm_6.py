
class Model(nn.Module):
    def forward(self, input_0, input_1, input_2, input_3, input_4, input_5, input_6):
        t0 = torch.matmul(input_0, input_1)
        t1 = torch.matmul(input_2, input_3)
        t2 = torch.matmul(input_4, input_5)
        t3 = torch.matmul(t0, t1)
        t4 = torch.matmul(t2, t3)
        t5 = t4 + input_6
        return t5
# Inputs to the model
input_0 = torch.randn(20, 20, requires_grad=True)
input_1 = torch.randn(20, 20, requires_grad=True)
input_2 = torch.randn(20, 20, requires_grad=True)
input_3 = torch.randn(20, 20, requires_grad=True)
input_4 = torch.randn(20, 20, requires_grad=True)
input_5 = torch.randn(20, 20, requires_grad=True)
input_6 = torch.randn(20, 20, requires_grad=True)
