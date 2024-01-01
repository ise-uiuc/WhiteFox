
class Model(torch.nn.Module):
    def forward(self, input):
        t1 = torch.mm(input, input) # Matrix multiplication between input and input
        t2 = torch.mm(input, input) # Matrix multiplication between input and input
        t3 = t1 + t2 # Addition of results of the two matrix multiplications
        t4 = t1 + t3 # Addition of results of the two matrix multiplications
        t5 = t1 - t4 # Subtraction of results of the two matrix multiplications
        return torch.mm(input, torch.mm(input, t5)) # Matrix multiplication of input and the last calculated term
# Inputs to the model
input = torch.randn(64, 64)
