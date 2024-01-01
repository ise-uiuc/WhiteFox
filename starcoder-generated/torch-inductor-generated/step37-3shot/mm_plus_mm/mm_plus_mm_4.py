
class Model(torch.nn.Module):
    def forward(self, tensorList, tensor2):
        t1 = torch.matmul(tensorList[1], tensorList[2])
        t2 = torch.matmul(tensorList[3], tensorList[4])
        t3 = torch.matmul(tensorList[5], tensorList[7])
        t4 = torch.matmul(tensorList[6], tensor2)
        t5 = t1 + t2
        t6 = t3 + t4
        t7 = t5 + t6
        return t7
# Inputs to the model
tensorList = [torch.randn(20, 20), torch.randn(20, 20), torch.randn(20, 20), torch.randn(20, 20), torch.randn(20, 20), torch.randn(20, 20),
              torch.randn(20, 20), torch.randn(20, 20)]
tensor2 = torch.randn(20, 20)
