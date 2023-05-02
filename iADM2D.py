import numpy as np
from Utils import *
def cconvfft2(A, B, *args):
    """ 2D Circular Convolution
    """
    numvararg = len(args)
    
    if numvararg > 2:
        raise Exception('Too many input arguments.')
    
    N = max(A.shape + B.shape)
    
    if numvararg >= 1 and args[0] is not None:
        N = args[0]
    
    A_hat = np.fft.fft2(A, s=(N, N))
    B_hat = np.fft.fft2(B, s=(N, N))
    
    if numvararg >= 2 and args[1] is not None:
        if args[1] == 'left':
            A_hat = np.conj(A_hat)
        elif args[1] == 'right':
            B_hat = np.conj(B_hat)
        elif args[1] == 'both':
            A_hat = np.conj(A_hat)
            B_hat = np.conj(B_hat)
        else:
            pass
    
    C = np.fft.ifft2(A_hat * B_hat)
   
    return np.real(C)
def innerprod(U,V):
    """ 2D Inner product
    """
    U_flat = U.flatten() # Flattens the array into a 1D array
    V_flat = V.flatten() # Flattens the array into a 1D array
    T = U_flat * V_flat
    f = T.sum()
    return f
def eval2D(Y,A,X):
    """ Objective Function 2D 
    """
    return 0.5*innerprod(Y-cconvfft2(A,X),Y-cconvfft2(A,X))
def eval12D(Y,A,X):
    t=X.flatten()
    return eval2D(Y,A,X)+np.linalg.norm(t,1)


def Proj(U, V):
    return V - np.sum(np.sum(np.conj(U) * V)) * U / np.linalg.norm(U.flatten())**2

def grad_X(Y,A,X):
    """ Gradient on the 2D objective function
    """
    Y_hat=cconvfft2(A,X)
    fx=Y_hat-Y
    n=len(Y)
    grad=np.real(cconvfft2(A,fx,n,'left'))  ###Correlate(Convolve)
    return 0.5*norm(fx)**2,grad
def grad_A(Y,A,X,case=True):
    Y_hat=cconvfft2(A,X)
    f=Y_hat-Y
    n=len(Y)
    grad=cconvfft2(X,f,n,'left')
    p1,p2=A.shape
    G=np.real(grad[:p1,:p2])
    if case :
        return 0.5*norm(f)**2,G
    else:
        return 0.5*norm(f)**2,Proj(A,G)

def backtracking2D(Y,A,X,f,gradf,lam,t):
    ### Line search by backtracking for the variable X
    
    Q = lambda Z,t : f+innerprod(gradf,Z-X) + (0.5/t)*norm(Z-X)**2
    t=8*t
  
    Xo=softh(X-t*gradf,lam*t)
    i=0
    while eval2D(Y,A,Xo) >  Q(Xo,t) :
    
        t=0.5*t
        Xo=softh(X-t*gradf,lam*t)
        i+=1
        if i > 100:
      
          break


    return Xo,t
def linesearch2D(Y,A,X,fa,grada):
  tau=0.9
  eta=0.8
  A1=Expomap(A,-tau*grada)
  
  i=0
  while eval2D(Y,A1,X) > fa - eta*tau*norm(grada)**2:
    tau=1/2*tau
    A1=Expomap(A,-tau*grada)
    i+=1
    if i > 100:
      break 
    
  return A1,tau
def iADM2D(Y,Ai,Xi,iter,lam,case=True,weight=1):

  A=Ai.copy()
  eps=0.001
  Ao=Ai
  X=Xi
  Xo=Xi
  lam=weight*lam
  t=1
  n=len(Y)
  L=[]
  for i in range(iter):
    
    beta=0.85
   
    
    X_hat=X+beta*(X-Xo)

    fx,gradfx=grad_X(Y,A,X_hat)
    
    
    Xo=X
    
    X,t=backtracking2D(Y,A,X_hat,fx,gradfx,lam,t)
    
    
    ### log map extension
    D=A-Ao
    #A_hat=Expomap(A,beta*D)
    A_hat=(A+beta*D)/norm(A+beta*D)
    

    
    
    fa,gradfa=grad_A(Y,A_hat,X,case)

    Ao=A
    A,_=linesearch2D(Y,A_hat,X,fa,gradfa)
    
    err=norm(cconvfft2(A,X)-Y)
    L.append(err)
    if err < 1e-2:
      break
    
  
  return A,X,L
def homotopy2D(Y,Ai,Xi,lam,lamf):
  eta=0.85
  N=int(np.log(lamf/lam)/np.log(eta))
  lam0=lam
  print(N)
  L=np.array([])
  
  for i in range(N):
   
    A,X,E=iADM2D(Y,Ai,Xi,100,lam0,False)
    Ai,Xi=A,X
    lam0=eta*lam0
    L=np.concatenate((L,np.array(E)))
    if L[99] < 1e-2:
       break
    print(i)
  return A,X,L
def reweighting2D(Y,Ai,Xi,lam,N):
  
 
  L=np.array([])
  m=len(Y)**2
  n=len(Ai)**2
  weight=1.
  
  for i in range(N):
   
    A,X,E=iADM2D(Y,Ai,Xi,100,lam,False,weight)
    Ai,Xi=A,X
    

    x = np.sort(np.abs(X.flatten()), axis=None)[::-1]
    thres = x[int(n / (4 * np.log(m/n)))]
    e = max(thres, 1e-3)
    weight= 1 / (np.abs(X) + e)

    L=np.concatenate((L,np.array(E)))
    if L[99] < 1e-2:
       break
    print(i)
  return A,X,L




def shift_correction_2D(A, X):
    
    n = [A.shape[0]//3, A.shape[1]//3]
    
    A_shift = np.zeros((n[0], n[1]))
    X_shift = np.zeros(X.shape)

    
    Corr = np.zeros((2*n[0], 2*n[1]))
    for i in range(3*n[0] - n[0]):
        for j in range(3*n[1] - n[1]):
            window = A[i:i+n[0], j:j+n[1]]
            Corr[i,j] = np.linalg.norm(window.flatten())
        
    max_val = np.max(Corr)
    ind_1, ind_2 = np.argwhere(Corr == max_val)[0]
        
    A_shift[:,:] = A[ind_1:ind_1+n[0], ind_2:ind_2+n[1]]
    
    X_shift[:,:] = np.roll(X[:,:], (ind_1+1, ind_2+1), axis=(0,1))
    
    return A_shift, X_shift
