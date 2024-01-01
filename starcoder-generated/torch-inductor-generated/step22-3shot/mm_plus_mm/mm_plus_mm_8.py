
class Model(torch.nn.Module):
    def forward(self, input1, input2, input3, input4):
        mm = torch.matmul(input1, input2).numpy()
        mm2 = torch.mm(input1, input2).numpy()
        mm3 = torch.mm(input3, input4).numpy()
        mm4 = torch.matmul(input3, input4).numpy()
        mm5 = np.dot(mm3, mm)
        mm6 = np.dot(input2.mm(input4), mm)
        return mm + mm2 + mm3
# Inputs to the model
input1 = torch.randn(5, 5)
input2 = torch.randn(5, 5)
input3 = torch.randn(5, 5)
input4 = torch.randn(5, 5)
