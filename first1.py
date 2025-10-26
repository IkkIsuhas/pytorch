import torch 
print('='*50)
print('PYTORCH-TUTORIALS PROGRAMS')
print('='*50)

# createing tensor from 1 to 6
x = torch.tensor([1,2,3,4,5,6])
print('ORIGINAL TENSOR',x)

reshaped = x.reshape(2,3)
print("RESHAPED TENSOR 2X3",reshaped)

rehaped2 = x.reshape(3,2)
print('RESHAPED TENSOR 3X2',rehaped2)