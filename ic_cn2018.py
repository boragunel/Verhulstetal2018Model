import numpy as np
from scipy import signal

M1=4.2767e-14 
M3=5.1435e-14 
M5=13.3093e-14 



##Standard
def cochlearNuclei(anfH,anfM,anfL,numH,numM,numL,fs):
    size=len(anfH[0,:])
    

    TF=19; #total no of fibers on each IHC
    # HSnormal=13;
    # MSnormal=3;
    # LSnormal=3;

    Acn=1.5;
    Scn=0.6;
    inhibition_delay=int(round(1e-2*fs));
    Tex=0.5e-3;                             #smoothed excitatory copy
    Tin=2e-3;                               #smoothed inhibitory copy

    summedAN=numL*anfL+numM*anfM+numH*anfH;

    summedAN=summedAN;
    delayed_inhibition=np.zeros_like(summedAN)
    delayed_inhibition[inhibition_delay:,:]=summedAN[0:len(summedAN)-inhibition_delay,:]

    #filters obtained with bilinear transform
    # # Excitatory filter:
    m = (2*Tex*fs)
    a = (m-1)/(m+1)
    bEx = 1.0/(m+1)**2*np.array([1,2,1]); #numerator
    aEx = np.array([1, -2*a, a**2]); #denominator

    # # Inhibitory filter:
    m = (2*Tin*fs)
    a = (m-1)/(m+1)
    bIncn = 1.0/(m+1)**2*np.array([1, 2, 1]); # numerator
    aIncn = np.array([1, -2*a, a**2]); # denominator

    cn=Acn*(signal.lfilter(bEx,aEx,summedAN,axis=0)-(Scn*signal.lfilter(bIncn,aIncn,delayed_inhibition,axis=0)))
    return cn,summedAN  ##SummedAN is wave I, cn is cochlear nuclei these are called in model 2018 to create wave I and III

##standard          
def inferiorColliculus(cn,fs):
    size=len(cn[0,:])
    Tex=0.5e-3;
    Tin=2e-3;
    Aic=1;
    Sic=1.5;
    inhibition_delay=int(round(2e-4*fs));

    delayed_inhibition=np.zeros_like(cn)
    delayed_inhibition[inhibition_delay:,:]=cn[0:len(cn)-inhibition_delay,:]

    # # Excitatory filter:
    m = (2*Tex*fs)
    a = (m-1)/(m+1)
    bEx=1.0/(m+1)**2*np.array([1,2,1]); #numerator
    aEx=np.array([1, -2*a, a**2]); #denominator

    # # Inhibitory filter:
    m = (2*Tin*fs)
    a = (m-1)/(m+1)
    bIncn = 1.0/(m+1)**2*np.array([1, 2, 1]); # numerator
    aIncn = np.array([1, -2*a, a**2]); # denominator

    ic=Aic*(signal.lfilter(bEx,aEx,cn,axis=0)-Sic*signal.lfilter(bIncn,aIncn,delayed_inhibition,axis=0)/2)
    return ic

def splitNuclei(summedAN,fs,cn=None,ic=None):
    
    Acn=1.5;
    Scn=0.6;
    Aic=1;
    Sic=1.5;
    Tex=0.5e-3;
    Tin=2e-3;

    def lowpass(T):                                     
        m = (2*T*fs)
        a = (m-1)/(m+1)
        return 1.0/(m+1)**2*np.array([1,2,1]), np.array([1, -2*a, a**2])

    def delayed(x,d):
        y=np.zeros_like(x)
        if d<len(x):
            y[d:]=x[:len(x)-d]
        return y

    bEx,aEx=lowpass(Tex)
    bIn,aIn=lowpass(Tin)

    def build(d_cn,d_ic):
        cn_pos=Acn*signal.lfilter(bEx,aEx,summedAN,axis=0)
        cn_neg=-Acn*Scn*signal.lfilter(bIn,aIn,delayed(summedAN,d_cn),axis=0)
        ic_pos=Aic*signal.lfilter(bEx,aEx,cn_pos,axis=0)
        ic_neg=Aic*(signal.lfilter(bEx,aEx,cn_neg,axis=0)
                    -Sic/2*signal.lfilter(bIn,aIn,delayed(cn_pos+cn_neg,d_ic),axis=0))
        return cn_pos,cn_neg,ic_pos,ic_neg

    if cn is not None and ic is not None:
        cn_pos,cn_neg,ic_pos,ic_neg=build(int(round(1e-2*fs)),int(round(2e-4*fs)))
        for name,rebuilt,model in (('cn',cn_pos+cn_neg,cn),('ic',ic_pos+ic_neg,ic)):
            if not np.allclose(rebuilt,model,rtol=1e-6,atol=1e-6*np.max(np.abs(model))):
                print(f"Warning: {name} rebuilt by splitNuclei differs from the model's {name}; the constants in "
                      f"splitNuclei no longer match cochlearNuclei / inferiorColliculus")
    return build(0,0)


