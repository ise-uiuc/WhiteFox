
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        v1 = x1.permute(0, 2, 1)
        v2 = x2.permute(0, 3, 1, 2)[:, 1]
        v3 = torch.bmm(x1.permute(0, 2, 1), x2.permute(0, 3, 1, 2)[:, 1])
        return v3
# Inputs to the model
x1 = torch.randn(1, 2, 2)
x2 = torch.randn(1, 4, 2, 2)
# Inputs end

# Model begins
#!/usr/bin/bash


for i in {0..4}
  do 
      for z in {0..4}
        do
            echo i = ${i}; 
            echo z = ${z}; 
            echo $(python./scripts/python/pytorch/permute-matmul.py 2 5 2 2 "${!i}" "${!z}"); 
        done
  done
