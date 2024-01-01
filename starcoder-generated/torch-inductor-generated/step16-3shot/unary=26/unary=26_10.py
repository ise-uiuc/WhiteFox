
a = torch.tensor(480.0)
b = torch.tensor(480)
a = a.type(torch.int)
print(a)
print(b)
if torch.equal(a, torch.tensor(b)):
    print('The two tensors are equivalent.')
else:
    print('The two tensors are not equivalent.')
