
import numpy as np
import math

# Integration imports
from scipy.integrate import quad
from scipy.ndimage import gaussian_filter
from scipy.integrate import simps
from scipy.integrate import dblquad


########################################### Constants of Nature ###################################


NA = 6.022*(10**23) # Avagadros constant N/mol

hbar = 6.582*(10**(-16))*(1.0/(10**6)) # MeV * seconds
c = 3*10**8 # m /second (speed of light of course)


GF = (1.166*(10**(-5)))*(1000**(-2)) # MeV^-2


# weak mixing angle contribution --> PDG value for now --> could vary
sin2thetaW = 0.231


# Some particle masses
m_mu = 105.66 # MeV
m_pi = 139.6 # MeV

#####################################################################################################




materials = {

	        # N, Z, Molar Mass, Mass of nucleus, 
	"Ar40": [
		22, # Number of Neutrons 
		18, # Number of Protons
		39.947*(1/1000), # Molar mass [kg/mol] 
		37227.56, # Mass of the nucleus in MeV --> Don't forget the binding energy!!!
	        3.427*10**-15 # Rmin in m for the Helm Form factor for Argon? --> Should double check this
		],


	"He":   [
		2, # Number of Neutrons 
		2, # Number of Protons
		4.0015*(1/1000), # Molar mass [kg/mol] 
		#28.3 , # Mass of the nucleus in MeV --> Don't forget the binding energy!!!
		3727.37, # Mass of the nucleus in MeV --> Don't forget the binding energy!!!
	        1.6755*10**-15 # Rmin in m for the Helm Form factor
	      	],

}



class SNSGen:

	def __init__(self, mdet, L, pot, material):
		self.mdet = mdet # detector mass in kg
		self.L = L # Distance from the source to the detector in meters
		self.pot = pot # The number of POT per year of operation or just the number of POT of interest
		self.material = material # the material to look up from the materials dict
		self.r = 0.08 # nu/POT/flavor at the SNS


		# apply the material properties chosen
		self.M = materials[material][3]
		self.Mmolar = materials[material][2]
		self.N = materials[material][0]
		self.Z = materials[material][1]
		self.Rmin = materials[material][4]

		self.Nt = self.mdet*(NA/self.Mmolar) # Number of targets
		self.Norm = self.pot*self.Nt # the exposure over the course of providing pot protons on target

	# KINEMATICS

	
	# Get the minimum required nu energy to produce a recoil of energy Er 
	def get_nu_min_energy(self, Er):
		return np.sqrt(self.M*Er/2)


	def get_min_cos(self, Er):
		A = 1.0/m_mu
		B = np.sqrt((self.M*Er)/2.)
		C = 2 + (m_mu/self.M)
		return A*B*C


	@staticmethod
	def spectral_nue(Enu):
		A = 192./m_mu
		B = (Enu/m_mu)**2
		C = 0.5 - (Enu/m_mu)
		S = (Enu <= (m_mu/2))
		return A*B*C*S

	
	@staticmethod
	def spectral_numubar(Enu):
		return 1.0/Enu


	@staticmethod
	def spectral_numubar(Enu):
		A = 64./m_mu
		B = (Enu/m_mu)**2
		C = 0.75 - (Enu/m_mu)
		maxv = m_mu/2
		S = (Enu <= maxv)
		return A*B*C*S


	# The numu spectral is mostly for show --> We will actually apply the delta function in any real calculations up front

	@staticmethod
	def spectral_numu(Enu):
    		V = (m_pi**2 - m_mu**2)/(2*m_pi)
    		dE = 0.001 #MeV
    		S1 = (Enu >= V - dE)
    		S2 = (Enu <= V + dE)
    		return ((2*m_pi)/(m_pi**2 - m_mu**2))*S1*S2



	def get_flux_norm(self):
		Nflux = self.r / (4*math.pi*(self.L**2)) # neutrinos / POT / Area
		return Nflux


	def flux_numu(self, Enu):
		A = (m_pi**2 - m_mu**2)/(2*m_pi)
		return self.spectral_numu(Enu)*self.get_flux_norm()*A

	def flux_numubar(self, Enu):
		return self.spectral_numubar(Enu)*self.get_flux_norm()


	def flux_nue(self, Enu):
		return self.spectral_nue(Enu)*self.get_flux_norm()


	def flux_muon(self, Enu):
		f1 = self.flux_nue(Enu)
		f2 = self.flux_numubar(Enu)
		return f1 + f2

	def flux_total(self, Enu):
		f1 = self.flux_nue(Enu)
		f2 = self.flux_numu(Enu)
		f3 = self.flux_numubar(Enu)
		return f1 + f2 + f3



	# Cross-section stuff 


	# Helm Form Factor
	def FH(self, Er):
		def j1(x):
			A = np.sin(x)/(x**2)
			B = np.cos(x)/x
			return A - B

		s = 0.9*10**(-15) # m
		Ro = np.sqrt((5./3)*(self.Rmin**2 - (3*s**2)))
		q = np.sqrt(2*self.M*Er)/(hbar*c)
		a = q*Ro
		A = 3*j1(a)/a
		B = -0.5*(q*s)**2
		return A*np.exp(B)



	def Qw(self):
		A = self.Z*(1 - 4*sin2thetaW)
		B = self.N - A
		return B



	def dSigma_dEr(self, Er, Enu):
		A = (GF**2)/(2*math.pi)
		B = (self.Qw()**2)/4.
		C = self.M*self.FH(Er)**2
		D = 2. - ((self.M*Er)/(Enu**2))
		E = (hbar*c)**2
		return A*B*C*D*E


	
	def Sigma(self, Enu):
		#def integrand(Er, Enu):
		#	return self.dSigma_dEr(Er, Enu)
        
		v = []
		for E in Enu:
			m = (2*E**2)/self.M
			I = quad(self.dSigma_dEr, 0, m, args=(E))

			v.append(I[0])

		return np.array(v)



	# differential rates / POT

	def rate_integrand_nue(self, Enu, Er):
		f = self.flux_nue(Enu)	
		sig = self.dSigma_dEr(Er, Enu)
		return f*sig

	def rate_integrand_numubar(self, Enu, Er):
		f = self.flux_numubar(Enu)	
		sig = self.dSigma_dEr(Er, Enu)
		return f*sig

	def dN_dEr_nue(self, Er):
		try:
			v = []
			for E in Er:
				a = self.get_nu_min_energy(E)
				b = m_mu/2
				I = quad(self.rate_integrand_nue, a, b, args=(E))
				v.append(I[0])
				return self.Nt*np.array(v)
		except:
			a = self.get_nu_min_energy(Er)
			b = m_mu/2
			I = quad(self.rate_integrand_nue, a, b, args=(Er))
			return self.Nt*I[0]    



	def dN_dEr_numubar(self, Er):
		try:
			v = []
			for E in Er:
				a = self.get_nu_min_energy(E)
				b = m_mu/2
				I = quad(self.rate_integrand_numubar, a, b, args=(E))
				v.append(I[0])
				return self.Nt*np.array(v)
		except:
			a = self.get_nu_min_energy(Er)
			b = m_mu/2
			I = quad(self.rate_integrand_numubar, a, b, args=(Er))
			return self.Nt*I[0]    


		

	def dN_dEr_numu(self, Er):
		Emono = (m_pi**2 - m_mu**2)/(2*m_pi)
		m = (2*Emono**2)/self.M
		f = self.get_flux_norm()
		sig = self.dSigma_dEr(Er, Emono)
		S = (Er <= m) 
		return self.Nt*f*sig*S



	# rates per POT

	def N_Er_nue(self, Er):
		dEr = Er[1] - Er[0]
		rs = []
		for E in Er:
			r = quad(self.dN_dEr_nue, E, E+dEr)[0]
			rs.append(r)
		return np.array(rs)


	def N_Er_numubar(self, Er):
		dEr = Er[1] - Er[0]
		rs = []
		for E in Er:
			r = quad(self.dN_dEr_numubar, E, E+dEr)[0]
			rs.append(r)
		return np.array(rs)


	def N_Er_numu(self, Er):
		dEr = Er[1] - Er[0]
		rs = []
		for E in Er:
			r = quad(self.dN_dEr_numu, E, E+dEr)[0]
			rs.append(r)
		return np.array(rs)


	# Nuclear Recoil Angles
	def epsilon(self, Er, costhetar):
		Enu_min = self.get_nu_min_energy(Er)
		return 1.0/((costhetar/Enu_min) - (1.0/self.M))


	# Nuclear Recoil Angles
	def dN_dEr_dcos_muon(self, Er, costhetar):
		min_cos = self.get_min_cos(Er)
		S1 = (costhetar > min_cos)
		Enu_min = self.get_nu_min_energy(Er)
		e = self.epsilon(Er, costhetar)
		S2 = (e >= Enu_min)
		S3 = (e <= (m_mu/2))
		sig = self.dSigma_dEr(Er, e)
		a = (e**2)/Enu_min
		f_nue = self.flux_nue(e)
		f_numubar = self.flux_numubar(e)
		return (self.Nt/(2*math.pi))*a*sig*(f_nue + f_numubar)*S1*S2*S3
		
























