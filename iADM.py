from Utils import grad1,Gradmat,eval2,backtracking,Logmap,Expomap,grad2,norm,linesearch,Conv
import numpy as np



def iADM(y,ai,xi,iter,lam):
  """
  The iADM algorithm for convolved signal recovery.
  
  Parameters:
    y: Convolved signal.
    ai: The initial guess for the signal vector.
    xi: The initial guess for the sparse coefficient vector.
    iter: The number of iterations for the algorithm.
    lam: The regularization parameter.
    
  Returns:
    a: The final estimate for the sparse coefficient vector.
    x: The final estimate for the signal vector.
    L: The list of errors.
  """
  
  a=ai.copy()           # Make a copy of the initial guess for a
  eps=0.001             # Set a small value for epsilon
  ao=ai*(1+eps)         # Define a small perturbation to a
  x=xi                  # Set x as the initial guess for the sparse coefficient vector
  xo=xi                 # Set xo as the initial guess for x
  
  t=1                   # Initialize the value of t
  n=len(y)              # Get the length of the input signal y
  L=[]                  # Create an empty list to store the errors
  for i in range(iter): # Run the loop for the specified number of iterations
   
    beta=0.85          # Set the value of beta
    
    ### Compute the extrapolated value of x
    x_hat=x+beta*(x-xo)
    
    ### Compute the gradient of f with respect to x
    G=Gradmat(a,n)
    gradfx=grad1(y,a,x_hat,G)
    
    ### Compute the value of f
    fx=eval2(y,a,x_hat)
    
    xo=x            ### update xo
    
    ### Compute the optimal value of x using backtracking line search
    x,t=backtracking(y,a,x_hat,fx,gradfx,lam,t)   
    
    ### Descent with respect to a
    D=a-ao
    a_hat=Expomap(a,beta*D)

    ### Compute the value of f with respect to a
    fa=eval2(y,a_hat,x)
    
    ### Compute the gradient of f with respect to a
    gradfa=grad2(y,a_hat,x)

    ao=a
    ### Compute the optimal value of a using line search
    a,_=linesearch(y,a_hat,x,fa,gradfa)
   
    ### Compute the error and append it to the list L
    err=norm(Conv(a,x)-y)
    L.append(err)
   
    
  return a,x,L





def homotopy(y,ai,xi,lam,lamf):
    
  """
  The iADM algorithm with Homotopy continuation for convolved signal recovery.
  
  Parameters:
    y: Convolved signal.
    ai: The initial guess for the signal vector.
    xi: The initial guess for the sparse coefficient vector.
    lamf: Final reguralization parametre
    lam: Initial regularization parameter.
    
  Returns:
    a: The final estimate for the sparse coefficient vector.
    x: The final estimate for the signal vector.
    L: The list of errors.
  """
  
  eta=0.85
  N=int(np.log(lamf/lam)/np.log(eta))
  lam0=lam
  print(N)
  L=np.array([])
  
  for i in range(N):
   
    a,x,E=iADM(y,ai,xi,100,lam0)
    ai,xi=a,x
    lam0=eta*lam0
    L=np.concatenate((L,np.array(E)))
  return a,x,L