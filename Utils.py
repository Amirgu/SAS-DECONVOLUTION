import numpy as np




def Conv(a,x):
  d=len(a)
  n=len(x)
  s=np.zeros(n)
  for i in range(n):
    for j in range(d):
      s[i]+=a[j]*x[i-j]
  return s

def softh(x,lam):
  v=np.abs(x)-lam*np.ones(len(x))
  
  u=np.multiply(np.sign(x),np.maximum(v,np.zeros(len(v))))
  return u

def Gradmat(a,n):
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

def Expomap(X,U):
 
  if norm(U) > 1e-3:
    return X*np.cos(norm(U))+(U/norm(U))*np.sin(norm(U))
  else:
    return (X+U) / norm(X+U)
  
def normalize(X):
  return X/norm(X)

def projT(z,x):
  return (np.eye(len(z))-np.outer(z,z)) @ x

def eval(y,a,x,lam):

  return lam*np.linalg.norm(x,1)+0.5*norm(Conv(a,x)-y)**2
def eval2(y,a,x):
  return 1/2*norm(Conv(a,x)-y)**2

def grad1(y,a,x,A):
  u=A.T@ (Conv(a,x)-y)
  return u

def Logmap(a,x):
  if norm(x-a) > 1e-3 and norm(a) == 1 : 
    u=projT(a,x-a)
    u=u/norm(u)
    v=np.arccos(a.T @ x)
    return u*v
  else:
    return x-a
  

def backtracking(y,a,x,f,gradf,lam,t):


    Q = lambda Z,t : f+ gradf @ (Z-x) + (0.5/t)*((Z-x)@(Z-x))
  
    t=8*t
  
    xo=softh(x-t*gradf,lam*t)
    i=0
    while eval2(y,a,xo) >  Q(xo,t) :
    
        t=0.5*t
        xo=softh(x-t*gradf,lam*t)
        i+=1
        if i > 100:
      
          break


    return xo,t

def grad2(y,a,x):
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
  tau=0.9
  eta=0.8
  a1=Expomap(a,-tau*grada)
  
  i=0
  while eval2(y,a1,x) > fa - eta*tau*norm(grada)**2:
    tau=1/2*tau
    a1=Expomap(a,-tau*grada)
    i+=1
    if i > 100:
      break 
    
  return a1,tau


def pad(y,p):
  s=np.pad(y[:p],(p,p),constant_values=0)
  return s


def BG(theta,n):

  x=np.random.binomial(1,theta,n)
  y=np.random.normal(0,1,n)
  z=x*y
  return z