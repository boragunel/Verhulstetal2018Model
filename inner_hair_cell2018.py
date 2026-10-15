



##comments on limitations of the model

##1) Solves the equations manually through Euler's integration or similar
##2) System is deterministic, only expected value considered, no Monte-Carlo
##3) To keep in mind that transduction dynamics are congruent across CF channels

from numpy import *
from scipy import signal
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

#parameters
Cm=12.5e-12;
Gmet=30e-9;
s1=16e-9;
s0=s1*3;
x0=20e-9;
#x0=34e-9
tauMet=50e-6;
Gleak=00.0e-9;
EP=90e-3;
Ekf=-71e-3;
Eks=-78e-3;
Gk=230e-9;
xk=-31e-3;
sk=10.5e-3;
tkf1=0.3e-3;
tkf2=0.1e-3;
tks1=8e-3;
tks2=2e-3;
GcaM=4.1e-9;
tauCa=0.25e-3;
xca=-30.0e-3;
sca=7.5e-3;
Eca=45e-3;
tauInS=0.5;
tauInF=50e-3;
CaInMax=0.4;
xCaIn=-43e-3;
sCaIn=6e-3;
Ileak=0.0e-9;
resting_potential=-0.05703; #resting_potential at equilibrium. This value is in Volts.
peak_potential=-0.04; #peak resting potential at 100 dB 4 kHz (where nerve fibers saturate)


###More efficient loop

def inner_hair_cell_potential(mu,fs,Gmet=Gmet,Gk=Gk,tauMet=tauMet):  ##reassign variables from function in for loop testing
    size=len(mu[0,:])   ##to remember mu is BM velocity, the value for this is inputted in the function in model2018. n_CF = 401, since mu has shape n_t x n_cf
    fs=fs               ##Sampling frequency of the stimulus         
    dt=1.0/fs
    alphaMet=exp(-dt/tauMet);
    alphakf1=exp(-dt/tkf1); 
    alphaks1=exp(-dt/tks1);

    zero_time=int(fs*50e-3)
    mu_zeros = zeros((len(mu[:,0])+zero_time, size))   # merged it to one loop to make it cleaner.
    mu_zeros[zero_time:,:] = mu                        #replaces the 0 in the list with the BM velocity input mu          
    mtIn=1/(1+exp((x0-mu_zeros)/s0)*(1+exp((x0-mu_zeros)/s1)))
    
    ##Variables initialised, updated in the loop
    mt=zeros([1,size])+1/(1+exp(x0/s0)*(1+exp(x0/s1))); ##shape = 1x401, Boltzman prob 
    Vsol=zeros_like(mu,dtype=float)         ##Takes the mu as input and convert each entry to float, shape = t_Nx401
    Vm=zeros([1,size])+resting_potential    ##Membrane potential across each channel expressed as a vector at t=0, has shape 1x401
    mkf1=1/(1+exp(-(Vm-xk)/sk));            ##Low pass filter, potassium fast channel      
    mks1=1/(1+exp(-(Vm-xk)/sk));            ##Low pass filter, potassium slow chnannel

    for i in range(len(mtIn[:,0])):
        mt=(1-alphaMet)*mtIn[i,:]+alphaMet*mt   ##mt is MET open probability
        Imet=(Gmet*mt)*(Vm-EP);                 ##conductance is scaled by mt, and multiplied by dV to get Imet
        Ileak=Gleak*(Vm-Eca);                   ##same case, difference in real V and steady state, multiplied by G to get I
        mk=1/(1+exp(-(Vm-xk)/sk));              ##
        mkf1=(1-alphakf1)*mk+alphakf1*mkf1;     ##
        mks1=(1-alphaks1)*mk+alphaks1*mks1;
        Gkf=Gk*mkf1;
        Gks=Gk*mks1;
        Ikf=Gkf*(Vm-Ekf);                        #Calculated with same ideas before
        Iks=Gks*(Vm-Eks);
        dV=-(Ileak+Imet+Ikf+Iks)/Cm;             ##uses kirchoff's law, throws I_m to other side, uses that to calculate difference in memebrane potential
        Vm=Vm+dV*dt;                             ##Euler's integration
        if i>=zero_time:
            Vsol[i-zero_time,:]=Vm
    # print(Vsol)
    return Vsol




#class ihc():
#    def __init__(self,cf,mu,fs):
#        size=len(cf)
#        self.cf=cf
#        self.fs=fs
#        dt=1/fs
#        self.alphaMet=exp(-dt/tauMet);
#        self.a=zeros([size,2]);
#        self.mt=zeros(size)
#        self.alphakf1=exp(-dt/tkf1);
#        self.alphaks1=exp(-dt/tks1);
#        self.Vm=zeros(size)+resting_potential
#        self.mt=zeros(size)+1/(1+exp(x0/s0)*(1+exp(x0/s1)));
#        self.Vsol=np.zeros_like(mu)
#        self.mtIn=1/(1+exp((x0-mu)/s0)*(exp((x0-mu)/s1)+1));
#        self.mkf=1/(1+exp(-(self.Vm-xk)/sk));
#        self.mks=1/(1+exp(-(self.Vm-xk)/sk));
#    
#    def solve_step(self):
#        fs=self.fs
#        dt=1/fs
#        zero_time=fs*200e-3;
#        for i in range(tpad):
#            Imet=(Gmet*mt)*(V-EP);
#            Ileak=Gleak*(V-Eca);
#            self.mk=1/(1+exp(-(self.Vm-xk)/sk));
#            self.mkf1=(1-self.alphakf1)*self.mk+self.alphakf1*self.mkf1;
#            self.mks1=(1-self.alphaks1)*self.mk+self.alphaks1*self.mks1;
#            Gkf=Gk*self.mkf1;
#            Gks=Gk*self.mks1;
#            Ikf=Gkf*(self.Vm-Ekf);
#            Iks=Gks*(self.Vm-Eks);
#            dV=-(Ileak+Imet+Ikf+Iks)/Cm;
#            self.Vm=self.Vm+dV*dt;
#        for i in range(len(self.mtIn[:,0])):
#            mt=(1-self.alphaMet)*self.mtIn[:,i]+self.alphaMet*self.mt
#            Imet=(Gmet*mt)*(V-EP);
#            Ileak=Gleak*(V-Eca);
#            self.mk=1/(1+exp(-(self.Vm-xk)/sk));
#            self.mkf1=(1-self.alphakf1)*self.mk+self.alphakf1*self.mkf1;
#            self.mks1=(1-self.alphaks1)*self.mk+self.alphaks1*self.mks1;
#            Gkf=Gk*self.mkf1;
#            Gks=Gk*self.mks1;
#            Ikf=Gkf*(self.Vm-Ekf);
#            Iks=Gks*(self.Vm-Eks);
#            dV=-(Ileak+Imet+Ikf+Iks)/Cm;
#            self.Vm=self.Vm+dV*dt;
#            self.Vsol[:,i]=self.Vm
#
#
#
#
#
#
#
#
