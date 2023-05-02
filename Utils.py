import numpy as np




def Conv(a,x):
  """ 1D Circular Convolution 
  """
  d=len(a)
  n=len(x)
  s=np.zeros(n)
  for i in range(n):
    for j in range(d):
      s[i]+=a[j]*x[i-j]
  return s

def softh(x, lam):
    """
    Soft thresholding operator.
    """
    
    # Calculate the v array by taking the absolute value of x, subtracting
    # lam times an array of ones with the same shape as x.
    v = np.abs(x) - lam * np.ones(x.shape)
    
    # Calculate the u array by taking the elementwise maximum of v and an array
    # of zeros with the same shape as v, multiplied elementwise by the sign of x.
    u = np.multiply(np.sign(x), np.maximum(v, np.zeros(v.shape)))
    
    # Return the u array.
    return u




def Gradmat(a,n):
  """
    Gradient of Convolution Function
  """
  t=np.zeros(n)
  t[:len(a)]=a
  a=t
  g=np.zeros((n,n))
  for i in range(n):
    for j in range(n):
      if 0 <= j <= i:
        g[i][j]=a[i-j]
      else:
        g[i][j]=a[n+i-j] 
  return g

def norm(x):
  return np.linalg.norm(x)

def Expomap(X, U):
    """
    Exponential map.
    """
    
    # Calculate the norm of U.
    U_norm = np.linalg.norm(U)
    
    if U_norm > 1e-3:
        # If the norm of U is greater than a small threshold (1e-3), apply the
        # exponential map formula.
        return X * np.cos(U_norm) + (U / U_norm) * np.sin(U_norm)
    else:
        # If the norm of U is less than or equal to the threshold, use a
        # simplified formula to avoid numerical issues.
        return (X + U) / np.linalg.norm(X + U)

  
def normalize(X):
  return X/norm(X)

def projT(z, x):
    """
    Projects x orthogonally to z.
    """
    
    # Calculate the outer product of z with itself.
    zzT = np.outer(z, z)
    
    # Calculate the projection of x onto the orthogonal complement of z by
    # subtracting the projection of x onto z from x.
    return (np.eye(len(z)) - zzT) @ x


def eval(y,a,x,lam):
  "Objective function with L1 Norm"
  return lam*np.linalg.norm(x,1)+0.5*norm(Conv(a,x)-y)**2
def eval2(y,a,x):
  "Objective function without the L1 norm"
  return 1/2*norm(Conv(a,x)-y)**2

def grad1(y, a, x, A):
    """
    Compute the gradient of the objective function with respect to X.
    
    Arguments:
    y -- the observed signal
    a -- the filter/kernel
    x -- the current estimate of the signal
    A -- the dictionary matrix
    
    Returns:
    u -- the gradient of the objective function with respect to X
    """
    
    # Compute the residual between the observed signal and the estimated 
    # signal convolved with the filter/kernel.
    res = Conv(a, x) - y
    
    # Compute the gradient of the objective function with respect to X using 
    # the residual and the transpose of the dictionary matrix.
    u = A.T @ res
    
    # Return the gradient.
    return u


def Logmap(a,x):
  " Log map "
  if norm(x-a) > 1e-3 and norm(a) == 1 : 
    u=projT(a,x-a)
    u=u/norm(u)
    v=np.arccos(a.T @ x)
    return u*v
  else:
    return x-a
  

def backtracking(y, a, x, f, gradf, lam, t):
    """
    Perform line search for variable X.
    
    Arguments:
    y -- the observed signal
    a -- the filter/kernel
    x -- the current estimate of the signal
    f -- the objective function value at the current estimate
    gradf -- the gradient of the objective function at the current estimate
    lam -- regularization parameter
    t -- step size parameter
    
    Returns:
    xo -- the updated estimate of the signal
    t -- the updated step size parameter
    """
    
    # Define a helper function that computes the objective function value for a given x.
    Q = lambda Z, t: f + gradf @ (Z - x) + (0.5 / t) * ((Z - x) @ (Z - x))
    
    # Update the step size parameter t.
    t = 8 * t
    
    # Compute the proximal operator of the soft thresholding function with the updated step size.
    xo = softh(x - t * gradf, lam * t)
    
    i = 0
    # While the objective function value for the updated estimate is greater than Q(xo, t),
    # update the step size t and compute the proximal operator of the soft thresholding function
    # with the updated step size. Repeat until convergence or a maximum number of iterations is reached.
    while eval2(y, a, xo) > Q(xo, t):
        t = 0.5 * t
        xo = softh(x - t * gradf, lam * t)
        i += 1
        if i > 100:
            break
    
    # Return the updated estimate of the signal and the updated step size.
    return xo, t


def grad2(y,a,x):
  "Gradient of objective function with respect to A"
  n=len(x)
  p=len(a)
  B=np.zeros((n,p))

  for i in range(len(x)):
    for j in range(len(a)):
      B[i][j]=x[i-j]
  
  g=grad1(y,a,x,B)

  g=projT(a,g)
  return g


def linesearch(y,a,x,fa,grada):
  "Linesearch for variable a "

  # Set the parameters for the linesearch
  tau=0.9 # Initial step size
  eta=0.8 # Fraction of decrease in the objective function

  # Compute a new estimate of a using the exponential map
  a1=Expomap(a,-tau*grada)

  i=0
  while eval2(y,a1,x) > fa - eta*tau*norm(grada)**2:
    # Decrease the step size tau if the decrease in objective function is not sufficient
    tau=1/2*tau 
    a1=Expomap(a,-tau*grada)
    i+=1
    if i > 100:
      break 

  # Return the updated value of a and the step size used
  return a1,tau



def pad(y,p):
  "padding operator"
  s=np.pad(y[:p],(p,p),constant_values=0)
  return s


def BG(theta,n):
  "Bernouilli-Gaussian Distribution with parametre theta of sparsity"
  x=np.random.binomial(1,theta,n)
  y=np.random.normal(0,1,n)
  z=x*y
  return z

def shiftrecovery(a0, a):
    """
    Recovers a shifted signal given the ground truth signal.
    Returns the correlation between the shifted signal and ground truth, 
    and the recovered signal.
    """
    
    # Compute the cross-correlation between the ground truth signal and the 
    # observed signal.
    Corr = np.correlate(a0, a, mode='full')
    
    # Find the index of the maximum correlation.
    ind = np.argmax(np.abs(Corr))
    
    # Extract the value of the maximum correlation.
    Corr_max = Corr[ind]
    
    # Determine the direction and magnitude of the shift based on the 
    # location of the maximum correlation.
    if ind - len(a0) >= 0:
        a_shift = np.sign(Corr_max) * np.roll(a, -(len(a0) - ind - 1))
    else:
        a_shift = np.sign(Corr_max) * np.roll(a, ind - len(a0) + 1)
    
    # Return the recovered signal and the correlation value.
    return a_shift, Corr_max * np.sign(Corr_max)


def coherence(a):
  "Coherence of the kernel a "
  L=[]
  for i in range(1,len(a)):
    L.append(np.abs(a@ np.roll(a,i)))
  return max(L)