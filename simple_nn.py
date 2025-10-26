import torch 
import torch.nn as nn
model = nn.Linear(3,2)
print('our tiny model: ',model)

input_data = torch.tensor([1.0,2.0,3.0])
output = model(input_data)
print('Input: ',input_data)
print('Output: ',output)
print("\nThe model learned some weights (multipliers) and biases (additions)!")
print("Weights:", model.weight)
print("Biases:", model.bias)