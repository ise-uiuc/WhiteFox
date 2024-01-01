
class Model(torch.nn.Module):
    def forward(self, tensor1, tensor2, tensor3, tensor4, tensor5):
        v1 = torch.mm(tensor1, tensor3)
        v2 = torch.mm(tensor2, tensor4)
        v3 = torch.mm(tensor5, tensor4)
        return v1 + v2 + v3
# Inputs to the model
tensor1 = torch.randn(5, 5)
tensor2 = torch.randn(5, 5)
tensor3 = torch.randn(5, 5)
tensor4 = torch.randn(5, 5)
