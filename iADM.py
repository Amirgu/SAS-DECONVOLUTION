from Utils import grad1,Gradmat,eval2,backtracking,Logmap,Expomap,grad2,norm,linesearch,Conv
import numpy as np

def iADM(y,ai,xi,iter,lam,reweighting=False):

  a=ai.copy()
  eps=0.001
  ao=ai*(1+eps)
  x=xi
  xo=xi
  
  t=1
  n=len(y)
  L=[]
  for i in range(iter):
    if reweighting:
      beta=(i-1)/(i+2)
    else:
      beta=0.85
   
    
    x_hat=x+beta*(x-xo)
    
    
   
    G=Gradmat(a,n)
    
    
    gradfx=grad1(y,a,x_hat,G)

    fx=eval2(y,a,x_hat)
    
    xo=x
    
    x,t=backtracking(y,a,x_hat,fx,gradfx,lam,t)
    
    
    ### log map extension
    D=a-ao
    a_hat=Expomap(a,beta*D)

    

    fa=eval2(y,a_hat,x)
    gradfa=grad2(y,a_hat,x)

    ao=a
    a,_=linesearch(y,a_hat,x,fa,gradfa)
   
    err=norm(Conv(a,x)-y)
    L.append(err)
   
    
  
  return a,x,L




def homotopy(y,ai,xi,lam,lamf):
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