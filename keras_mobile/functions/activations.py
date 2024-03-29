import keras.backend as K

def Swish(beta = 1.0):
  def stub(x):
      return K.sigmoid(x * K.constant(beta)) * x
  return stub

def Mish():
  def stub(x):
    return x*K.tanh(K.softplus(x))
  return stub

def log_softmax(x, dim=-1):
  return x - K.logsumexp(x, axis=dim)
