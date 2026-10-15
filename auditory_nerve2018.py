from numpy import *
from inner_hair_cell2018 import resting_potential, peak_potential
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy import sparse

#parameters
r1=220. #reserve pool max. replenishment rate
x=700 #RRP replenishment rate (Pangrisc 2010,Chapocnikov 2014)
M=14 # Max. vesicles in the ready release pool or release sites (reasonable, ref?)
M2=60 # Max. vesicles in the second pool, fitted parameter
# with the paramaters is 250 release/s, because of refractoriness the steady state spike rate goes around 200 spike/s
refr_tail=0.6e-3 # relative refreactory period (from Peterson and Heil 2014)
abs_refractoriness=0.6e-3 # absolute refractory period (from Peterson and Heil 2014)
ss=1.5e-3 #sensitivity of release rate on IHC potential, fitted paramter
tCa=0.2e-3 #time constant of the Ca2 channel
#driven exocytosis rate is a boltzmann version of Vm, low-pass filter with the time constant of the Ca2+ channels. This is a shortcut to tune the nonlinear relationship between Vm and exocytosis rate based on AF data. Ca2+ signaling varies substantialy between synapses of the same IHC (Frank 2009,Ohn 2016). We use a shortcut Vm->Exocytosis rate, otherwise Vm->Ca2+ (at individual synapse)->exocytosis rate would require the tuning of too many parameters (here one free parameter ss, and 2 fitted parameters sp,psr)

'''
def auditory_nerve_fiber(Vm,fs,spont,r1=r1,x=x,M=M,M2=M2,tCa=tCa,ss=ss):  ##reassign variables from function in for loop testing
    size=len(Vm[0,:])   ##Vm is the IHC receptor potential from inner_hair_cell2018, shape n_t x n_cf (401 channels)
    dt=1.0/fs
    if(spont==0):
        sp=1 #spont rate
        psr=800 #peak spike rate, based upon  Taberner and Liberman 2005
    elif(spont==1):
        sp=10.0
        psr=1.0e3
    else:
        sp=68.5
        psr=3.0e3
    #parameters for vesicles trafficking
#    psr=2.7e3
#    psr=psr*4;
    r=r1/M2 # replenishment rate per vesicle location
    M2_steady=M2-sp/r #steady state values
    Msteady=M*(M2_steady/M2-sp/x)
    wt=M2_steady+zeros([1,size]) #reserve pool
    qt=Msteady+zeros([1,size]) #RRP
    xdt=x*dt
    rdt=r*dt
    #parameters for refractoriness
    alpha_ref=exp(-dt/refr_tail)
    relRefFraction=zeros([1,size]) #how much the firing probability decreases due to relative refractoriness
    available=1.0+zeros([1,size]) #number of fibers not in a refractory state
    buf_lgt=int(round(abs_refractoriness*fs)) #length of buffer to store the firing history (in order to account for absolute refractoriness)
    buf_ref=zeros([buf_lgt,size],dtype=float) #buffer to store the history of firing
    buf_pointer=0
    pp=psr/Msteady#/(1/(1+exp(-(peak_potential-vh)/ss))) #multiplier relating the activation nonlinearity (between 0 and 1) with actual firing rate
    #parameters to relate Vm with firing rate
    rat=log((psr-sp)/sp) #find half activation voltage of exocytosis rate based on peak and spontaneous rate
    vh=rat*ss+resting_potential
    k0=sqrt(1/(1+exp(-(resting_potential-vh)/ss)))+zeros([1,size])  #driven exocytosis rate at rest
    #take the square root, filter it with a first order filter and then square it. This is equivalent to a second order activation of the ion channels


    zero_time=int(50e-3*fs)                 ##50 ms settling window, expressed in samples
    Vm_zeros = zeros((len(Vm[:,0])+zero_time, size)) + resting_potential  # merging into one loop
    Vm_zeros[zero_time:,:] = Vm                        #replaces the resting values with the IHC potential input Vm
    kin=sqrt(1/(1+exp(-(Vm_zeros-vh)/ss)))             ##during the settling part kin==k0, so the Ca filter leaves k at k0
    solution=zeros_like(Vm)                 ##output firing, shape = n_t x 401 (sized to the REAL input, not the padded one)
    t_grid=arange(len(kin[:,0]))*dt         ##solve_ivp needs an interpolant, kin is only sampled
    n_t=len(kin[:,0])
    def kin_at(t):                          ##straight line between samples, same as interp1d but much cheaper
        j=int(t/dt)
        if j>n_t-2:
            j=n_t-2
        f=t/dt-j
        return kin[j]+f*(kin[j+1]-kin[j])

    def rhs(t,y):                           ##states: k (Ca filter), qt (RRP), wt (reserve)
        k,qt,wt = y[0:size],y[size:2*size],y[2*size:3*size]
        repRRP=x*fmax(wt/M2-qt/M,0)         ##three-store diffusion, reserve -> RRP
        return concatenate([(kin_at(t)-k)/tCa,       ##dk   (was the alphaCa filter)
                            repRRP-pp*(k**2)*qt,     ##dqt  exocytosis depletes the RRP
                            -repRRP+r*(M2-wt)])      ##dwt  implicit global store -> reserve
    y0=concatenate([k0.ravel(),qt.ravel(),wt.ravel()])
    sol=solve_ivp(rhs,(t_grid[0],t_grid[-1]),y0,t_eval=t_grid,
              method='RK45',rtol=1e-3,atol=1e-6,max_step=dt)
    
    k_all,qt_all = sol.y[0:size,:].T, sol.y[size:2*size,:].T
    ejected_all = pp*(k_all**2)*qt_all*dt   ##vesicles released per sample

    for i in range(len(t_grid)):            ##refractoriness only. buf_ref is a delay line,
        firing=(available-relRefFraction)*ejected_all[i,:]   ##so this part cannot go in solve_ivp
        recovered=buf_ref[buf_pointer,:]
        relRefFraction=alpha_ref*relRefFraction+(1-alpha_ref)*recovered
        available=available-firing+recovered
        buf_ref[buf_pointer,:]=firing
        buf_pointer=mod(buf_pointer+1,buf_lgt)
        if i>=zero_time:                        ##only start recording once the settling part is done
            solution[i-zero_time,:]=firing
    return solution                             ## Output shape is t_N x 401




'''
def auditory_nerve_fiber(Vm,fs,spont,r1=r1,x=x,M=M,M2=M2,tCa=tCa,ss=ss):  ##reassign variables from function in for loop testing
    size=len(Vm[0,:])   ##Vm is the IHC receptor potential from inner_hair_cell2018, shape n_t x n_cf (401 channels)
    dt=1.0/fs
    if(spont==0):
        sp=1 #spont rate
        psr=800 #peak spike rate, based upon  Taberner and Liberman 2005
    elif(spont==1):
        sp=10.0
        psr=1.0e3
    else:
        sp=68.5
        psr=3.0e3
    #parameters for vesicles trafficking
#    psr=2.7e3
#    psr=psr*4;
    r=r1/M2 # replenishment rate per vesicle location
    M2_steady=M2-sp/r #steady state values
    Msteady=M*(M2_steady/M2-sp/x)
    wt=M2_steady+zeros([1,size]) #reserve pool
    qt=Msteady+zeros([1,size]) #RRP
    xdt=x*dt
    rdt=r*dt
    #parameters for refractoriness
    alpha_ref=exp(-dt/refr_tail)
    relRefFraction=zeros([1,size]) #how much the firing probability decreases due to relative refractoriness
    available=1.0+zeros([1,size]) #number of fibers not in a refractory state
    buf_lgt=int(round(abs_refractoriness*fs)) #length of buffer to store the firing history (in order to account for absolute refractoriness)
    buf_ref=zeros([buf_lgt,size],dtype=float) #buffer to store the history of firing
    buf_pointer=0
    pp=psr/Msteady#/(1/(1+exp(-(peak_potential-vh)/ss))) #multiplier relating the activation nonlinearity (between 0 and 1) with actual firing rate
    #parameters to relate Vm with firing rate
    rat=log((psr-sp)/sp) #find half activation voltage of exocytosis rate based on peak and spontaneous rate
    vh=rat*ss+resting_potential
    k0=sqrt(1/(1+exp(-(resting_potential-vh)/ss)))+zeros([1,size])  #driven exocytosis rate at rest
    #take the square root, filter it with a first order filter and then square it. This is equivalent to a second order activation of the ion channels
    alphaCa=exp(-dt/tCa)
 

    zero_time=int(50e-3*fs)                 ##50 ms settling window, expressed in samples
    Vm_zeros = zeros((len(Vm[:,0])+zero_time, size)) + resting_potential  # merging into one loop
    Vm_zeros[zero_time:,:] = Vm                        #replaces the resting values with the IHC potential input Vm
    kin=sqrt(1/(1+exp(-(Vm_zeros-vh)/ss)))             ##during the settling part kin==k0, so the Ca filter leaves k at k0
    k=k0
    solution=zeros_like(Vm)                 ##output firing, shape = n_t x 401 (sized to the REAL input, not the padded one)
    for i in range(len(kin[:,0])):          ##From this line to 63 and line 68, exocytosis
        k=alphaCa*k+(1-alphaCa)*kin[i,:]    #From line 69 to 72 you have three-store diffusion
        vesicleReleaseRate=pp*(k**2)        
        releaseProb=vesicleReleaseRate*dt
        qt[qt<0]=0
        wt[wt<0]=0
        qt[qt>M]=M
        wt[wt>M2]=M2
        ejected=releaseProb*qt
        replenishReserve= rdt*(M2-wt)       ##This is where you have the euler integration
        replenishRRP=xdt*fmax(wt/M2-qt/M,0) ##This is where you have the euler integration
        qt= qt + replenishRRP - ejected
        wt= wt - replenishRRP + replenishReserve
        firing=(available-relRefFraction)*ejected
        recovered=buf_ref[buf_pointer,:]
        relRefFraction=alpha_ref*relRefFraction+(1-alpha_ref)*recovered
        available=available-firing+recovered
        buf_ref[buf_pointer,:]=firing
        buf_pointer=mod(buf_pointer+1,buf_lgt)
        if i>=zero_time:                        ##only start recording once the settling part is done
            solution[i-zero_time,:]=firing      
    return solution                             ## Output shape is t_N x 401
