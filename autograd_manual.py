import numpy as np

x = np.array([1,2,3,4],dtype=np.float32)
y = np.array([2,4,6,8],dtype=np.float32)

w = 0.0

# Model preduction
def forward(x):
  return w * x

# Loss
def loss(y,y_predicted):
  return((y_predicted - y)**2).mean()

# Gradient
# MSE = 1/N*(w*x-y)**2
# DJ/DW = 1/N 2X(w**x-y)
def gradent(x,y,y_predicted):
  return np.dot(2 * x,y_predicted - y).mean()
print(f'Prediction before training: f(5) = {forward(5):.3f}')

learning_rate = 0.01
n_iters = 20

for epoch in range(n_iters):
  y_pred = forward(x)
  l = loss(y,y_pred)
  dw = gradent(x,y,y_pred)

  w -= learning_rate * dw

  if epoch % 2 == 0:
    print(f'epoch {epoch+1}: w = {w:.3f}, loss = {l:.8f}')

print(f'Prediction after training: f(5) = {forward(5):.3f}')