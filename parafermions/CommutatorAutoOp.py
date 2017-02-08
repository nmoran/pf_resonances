from MPO import MPO
import time
import numpy as np
import scipy as sp

class CommutatorAutoOp(MPO):
    """
    Class for automatic commutator which makes a commutator
    from a regular MPO automatically.
    """

    def __init__(self, existing):
        """
        Constructor of auto commutator MPO class.

        Parameters
        ------------
        existing: MPO object
            MPO object for operator to turn into a commutator.
        """
        e = existing
        L = self.L = e.L
        N = self.N = e.N**2
        chi = self.chi = 2*e.chi
        dtype = self.dtype = e.dtype

        Ws = {}

        H1 = np.zeros((1,M,d,d), dtype=dtype)   #H1 = zeros(1,d,M,d);
        H1[0,0,:,:] = I                         #H1(1,:,1,:) = I;
        H1[0,1,:,:] = -Y                        #H1(1,:,2,:) = -Y;
        H1[0,2,:,:] = X                         #H1(1,:,3,:) = X;
        H1[0,3,:,:] = -Y*r[0]                   #H1(1,:,4,:) = -Y*r(1);
        H1[0,4,:,:] = X*r[0]                    #H1(1,:,5,:) = X*r(1);
        H1[0,11,:,:] = Pw*Z                     #H1(1,:,12,:)=Pw*Z;
        H1[0,12,:,:] = -Nw[0,0]*Z/2.0             #H1(1,:,13,:)= -Nw(1)/2*(Z);
        Ws[0] = H1                              #H{1,1} = (H1);

        for n in range(1, 2*N-1):                   #for n=2:2*N-1
            Hn = np.zeros((M,M,d,d), dtype=dtype)   #Hn = zeros(M,d,M,d);
            Hn[0,0,:,:] = I                         #Hn(1,:,1,:) = I;
            Hn[0,1,:,:] = -Y                        #Hn(1,:,2,:) = -Y;
            Hn[0,2,:,:] = X                         #Hn(1,:,3,:) = X;
            Hn[0,3,:,:] = -Y*r[n]                   #Hn(1,:,4,:) = -Y*r(n);
            Hn[0,4,:,:] = X*r[n]                    #Hn(1,:,5,:) = X*r(n);
            Hn[11,11,:,:] = Z                       #Hn(12,:,12,:)=Z;

            Hn[0,12,:,:] = -Nw[0,n]*(Z-I)/2.0         #Hn(1,:,13,:) = -Nw(n)/2*(Z-I);

            Hn[1,5,:,:] = X                         #Hn(2,:,6,:) = X;
            Hn[1,6,:,:] = Z                         #Hn(2,:,7,:) = Z;
            Hn[2,7,:,:] = Z                         #Hn(3,:,8,:) = Z;
            Hn[2,8,:,:] = Y                         #Hn(3,:,9,:) = Y;
            Hn[3,12,:,:] = X                        #Hn(4,:,13,:) = X;
            Hn[4,12,:,:] = Y                        #Hn(5,:,13,:) = Y;
            Hn[5,9,:,:] = Y*u[n]                    #Hn(6,:,10,:) = Y*u(n);%;1i*sy;
            Hn[6,9,:,:] = p[n]*Z                    #Hn(7,:,10,:) = p(n)*Z;
            Hn[7,10,:,:] = p[n]*Z                   #Hn(8,:,11,:) = p(n)*Z;
            Hn[8,10,:,:] = X*u[n]                   #Hn(9,:,11,:) = X*u(n);
            Hn[9,12,:,:] = X                        #Hn(10,:,13,:) = X;
            Hn[10,12,:,:] = Y                       #Hn(11,:,13,:) = Y;%;1i*sy;
            Hn[12,12,:,:] = I                       #Hn(13,:,13,:) = I;
            Ws[n] = Hn

        HN = np.zeros((M,1,d,d), dtype=dtype)       #HN = zeros(M,d,1,d);
        HN[0,0,:,:] = -Nw[0,2*N-1]/2*(Z-I)+Pw*I       #HN(1,:,1,:) = -Nw(2*N)/2*(Z-I)+Pw*I;% Only put in -I when using zero Nw at site 1
        HN[3,0,:,:] = X                             #HN(4,:,1,:) = X;
        HN[4,0,:,:] = Y                             #HN(5,:,1,:) = Y;
        HN[9,0,:,:] = Z                             #HN(10,:,1,:) = X;
        HN[10,0,:,:] = Y                            #HN(11,:,1,:) = Y;
        HN[12,0,:,:] = I                            #HN(13,:,1,:) = I;
        HN[11,0,:,:] = Z                            #HN(12,:,1,:)=Z;

        Ws[2*N-1] = HN

        self.Ws = Ws
        self.Lp = np.ones(1, dtype=dtype)
        self.Rp = np.ones(1, dtype=dtype)




