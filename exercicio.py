import numpy as np
import matplotlib.pyplot as plt

def polApprox(abscissa, ordinate, polDegree):
	M = np.zeros(shape=(polDegree+1,polDegree+1))
	b = np.zeros(polDegree+1)
	
	# Recursive power
	def powSum(base,exponent,factor):
		p = pow(base,exponent)
		pSum = 0.
		for i in xrange(np.size(p)):
			pSum = (p[i] * factor[i]) + pSum
		return pSum
	
	# Calculate M
	for j in xrange(polDegree+1):
		for i in xrange(polDegree+1):
			M[i,j] = powSum(abscissa,j+i,np.ones(np.size(abscissa)))

	# Calculate b
	for j in xrange(polDegree+1):
		b[j] = powSum(abscissa,j,ordinate)
		
	a = np.linalg.solve(M,b)

	return a

# Polynomial fitting function calculator
def f(abscissa, A):
	ordinate = 0
	for i in xrange(np.size(A)):
		ordinate = A[i] * pow(abscissa,i) + ordinate
	return ordinate

modelName = "./slab/slab.-1400"
lon,x,y = np.loadtxt(modelName,unpack=True)

a = polApprox(x,y,8)

yy = f(x,a)

plt.plot(x,y,'k--')
plt.plot(x,yy,color="blue")

plt.show()


