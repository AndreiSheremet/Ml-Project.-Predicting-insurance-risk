import numpy as np
def relu(x):
  return np.maximum(0,x)  #Turns negative into 0, and positive into itself

def relu_deriv(x):          #Derivative =1 if x>0 , else0
 return (x>0).astype(float)

def mse(y_true, y_pred):        #Mean squared error
 return np.mean((y_true - y_pred)**2) 

def mse_grad(y_true, y_pred):       #Derivative of mean squared error loss
  return 2 * (y_pred - y_true) / y_true.size


class FFNN:
    def __init__(self,input_size, hidden_sizes, output_size=1, lr=0.01, seed=31):
        np.random.seed(seed)
        self.lr=lr
        self.weight=[]
        self.biases=[]

        prev= input_size
        #Create hidden layers
        for h in hidden_sizes:
            W=np.random.randn(prev,h) * np.sqrt(2.0/ prev) #small random value
            b=np.zeros((1,h))       #biases start at zero
            self.weights.append(W)
            self.biases.append(b)
            prev=h
        #Create output layer
        W=np.random.randn(prev,output_size) * np.sqrt(2.0/ prev)
        b=np.zeros((1, output_size))
        self.weights.append(W)
        self.biases.append(b)
    
    def forward(self,X):
       activations= [X]
       preacts =[]

       for i in range(len(self.weights) -1):
          z= activations[-1] @ self.weights[i] + self.biases[i]
          preacts.append(z)
          a =relu(z)
          activations.append(a)
       z = activations[-1] @ self.weights[-1] +self.biases[-1]
       preacts.append(z)
       activations.append(z)

       return activations, preacts
    
    def backward(self,activations,preacts, y_true):
       grads_w, grads_b= [],[]
       y_pred =activations[-1]
       dY = mse_grad(y_true, y_pred)   # error from output

       #Output layer gradient
       grad_w= activations[-2].T @  dY
       grad_b=np.sum(dY,axis=0, keepdims=True)
       grads_w.insert(0, grad_w)
       grads_b.insert(0, grad_b)

       #Backprop through hidden layers
       dA =dY @ self.weights[-1].T
       for i in reversed(range(len(self.weights) -1)):
          dZ =dA * relu_deriv(preacts[i])
          grad_w = activations[i].T @ dZ
          grad_b = np.sum(dZ, axis=0, keepdims=True)
          grads_w.insert(0, grad_w)
          grads_b.insert(0, grad_b)
          if i != 0:
              dA = dZ @ self.weights[i].T

        # Update weights and biases
       for i in range(len(self.weights)):
            self.weights[i] -= self.lr * grads_w[i]
            self.biases[i] -= self.lr * grads_b[i]
       return mse(y_true, y_pred)
       
    def fit(self, X, y, epochs=1000):
        for epoch in range(epochs):
            activations, preacts = self.forward(X)
            loss = self.backward(activations, preacts, y)
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.6f}")

    def predict(self, X):
        a, _ = self.forward(X)
        return a[-1]
    


