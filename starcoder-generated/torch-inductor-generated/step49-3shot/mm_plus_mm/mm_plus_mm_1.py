
class Model(torch.nn.Module):
    def forward(self, tensor1, tensor2):
        t1 = torch.mm(tensor1, tensor1)
        t2 = torch.mm(tensor2, tensor2)
        t3 = torch.mm(t1, t1)
        t4 = torch.mm(t2, t2)
        t5 = torch.mm(t3, t3)
        t1 = t3 + t5
        t6 = torch.mm(t4, t4)
        t2 = t6 + t1
        return t3 + t6 + t1 + t2
# Inputs to the model
tensor1 = torch.randn(100, 100)
tensor2 = torch.randn(100, 100)
