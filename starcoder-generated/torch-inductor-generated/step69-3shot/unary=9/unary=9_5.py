
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = torch.nn.Conv2d(3, 8, (1, 6), stride=1, bias=True)
        self.b = torch.nn.Conv2d(8, 1, 1, stride=1, bias=True)
        self.c = torch.nn.Conv2d(8, 2, (1, 6), stride=1, bias=True)
        self.d = torch.nn.Conv2d(2, 4, 1, stride=1, bias=True)
        self.e = torch.nn.Conv2d(4, 5, (1, 6), stride=1, bias=True)
    def forward(self, x1):
        v1 = self.a(x1)
        v2 = self.c(v1)
        v3 = self.e(v2)
        v4 = v1.add(3)
        v5 = v2.add(3)
        v6 = v3.add(3)
        v7 = v4.add(torch.clamp_min(v5, -1))
        v8 = v6.add(torch.clamp_min(torch.clamp_max(v7, -1), 1))
        v9 = v8.div(6)
        result = self.b(v9)
        v10 = self.d(result)
        v11 = v10.add(3)
        return v11.clamp(min=0)
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
# Model end

# Description of the input tensor(s)

## Description of how the input tensor(s) was/were generated (random seed was applied if available)

# Description of the output dimensions for the model

# Description of the operations applied to the model
## Description of the different patterns that were explored and why they met the requirements
## Descripion of operations that are not part of the pattern other than to meet requirements
## Description of the order in which operations were applied and why they met the requirements
## Please describe the reason the input tensor(s) to the model need to be in that format/shape, such as channel first if thats needed.
## Please describe how the input tensor(s) was resized if it had to be. For example, the input size of a CNN is often larger than it needs to be to perform the models specified operations. Please describe what algorithm was applied to achieve this and why it was necessary.
## Please briefly decribe the pre-processing steps that were run on the models input if any.

# Example input(s) & output(s)
