# MATH 496/696 Spring 2016 Project 1
#
# Copyright (c) 2016 by Michael Robinson <michaelr@american.edu>
# Permission is granted to copy and modify for educational use, provided that this notice is retained
# Permission is NOT granted for all other uses -- please contact the author for details

# Load the necessary modules
import math
import numpy as np

# Remove the key when you're done writing!
import pr1_key

# Given function definitions
def kernel(A,tol=1e-5):
	"""Return a matrix whose columns span the kernel of the matrix A"""
	u, s, vh = np.linalg.svd(A)
	sing=np.zeros(vh.shape[0],dtype=np.complex)
	sing[:s.size]=s
	null_mask = (sing <= tol)
	null_space = np.compress(null_mask, vh, axis=0)
	return null_space.conj().T

def cokernel(A,tol=1e-5):
	u, s, vh = np.linalg.svd(A)
	sing=np.zeros(u.shape[1],dtype=np.complex)
	sing[:s.size]=s
	null_mask = (sing <= tol)
	return np.compress(null_mask, u, axis=1)

# Copyright (c) 2016 by Brian DiZio <brian.a.dizio@gmail.com>
# Permission is NOT granted for all other uses -- please contact the author for details

def isChainComplex(chaincomplex,tol=1e-5 ):
	"""Is a list of matrices a chain complex?"""
	for i in range(len(chaincomplex)):
		# If it is the first nonzero map, compose the first map with the zero map
		if i==0:
			A = np.dot(chaincomplex[i],np.zeros((chaincomplex[i].shape[1],chaincomplex[i].shape[1]))) # Composition
			for j in range(len(A)): # For every row
				for k in range(len(A[0])): # For the entries in each row
					if A[j][k]>tol: # Compare entries to tolerance
						#print "No chain complex" 
						return False
		elif len(chaincomplex)>i>0: # Compose the next map with the previous map not including 0th=0 map
			A = np.dot(chaincomplex[i],chaincomplex[i-1]) # Composition
			for j in range(len(A)): # For every row
				for k in range(len(A[0])): # For the entries in each row
					if abs(A[j][k])>tol: # Compare entries to tolerance
						#print "No chain complex"
						return False
	else:
			#print("The map is a chain complex")
			return True
	return isChainComplex(chaincomplex,tol=1e-5 )

def isExact(chaincomplex,tol=1e-5 ):
	"""Is a chain complex exact?"""
	for i in range(len(chaincomplex)): 
		if i==0: # If it is the first map, its kernel to the image the zero map
			b = kernel(chaincomplex[0])
			A = np.zeros((b.shape[0],1))
			for i in range(b.shape[1]):
				x = np.linalg.lstsq(A,b[:,i]) 
				if np.linalg.norm(np.dot(A,x[0])-b[:,i])>tol:
					#print("The complex is not exact.")           
					return False
		elif len(chaincomplex)>=i>0: # Compare next kernel to previous image, starting with second map to first map ending with last map to second to last map
			b = kernel(chaincomplex[i])
			A = chaincomplex[i-1]
			for i in range(b.shape[1]):
				x = np.linalg.lstsq(A,b[:,i])
				if np.linalg.norm(np.dot(A,x[0])-b[:,i])>tol:
					#print("The complex is not exact.")           
					return False
		elif i==len(chaincomplex):
			A=chaincomplex[i-1]
			b=kernel(np.zeros(A.shape))
			for i in range(b.shape[1]):
				x = np.linalg.lstsq(A,b[:,i])
				if np.linalg.norm(np.dot(A,x[0])-b[:,i])>tol:
					#print("The complex is not exact.")           
					return False
	else:
		#print("The complex is exact")
		return True

def homology_pairwise(d1,d2,tol=1e-5):
	"""Compute the homology of a pair of maps U -d1-> V -d2-> W"""
	# print "d1"
	# print d1
	# print "kernel(d2)"
	# print kernel(d2)
	b=kernel(d2)
	global homology_set
	homology_set=[]
	for i in range(b.shape[1]):  # for every column in the kernel
	  	x = np.linalg.lstsq(d1,b[:,i]) 
	  	if np.linalg.norm(np.dot(d1,x[0])-b[:,i])>=tol: #check if it is a linear combo of the columsn
	  		ap=b[:,i].tolist()
	  		homology_set.append(ap)
	if len(homology_set)==0:
		homology_array=np.zeros((d2.shape[0],0))
	else:
		homology_array=np.array(homology_set)
		homology_array=homology_array.T
	return homology_array

	# global homology_pair
	# m=np.linalg.lstsq(kernel(d2),d1)
	# print "cokernel(m[0])"
	# print kernel(m[0])
	# homology_pair=kernel(cokernel(m[0]))
	# print "homology_pair"
	# print homology_pair
	# return homology_pair

def homology(chaincomplex,k,tol=1e-5 ):
	"""Compute the homology of a chain complex, returning a matrix whose columns are a basis for the homology space at the desired degree"""
	if k==0:
		return kernel(chaincomplex[k])
	elif len(chaincomplex)-1>=k:
		return homology_pairwise(chaincomplex[k-1],chaincomplex[k])
	elif k==len(chaincomplex):
		d_end=np.zeros((1,chaincomplex[k-1].shape[0]))
		return homology_pairwise(chaincomplex[k-1],d_end)
	else:
		return np.array([])

def isChainMap(chaincomplex1,chaincomplex2,maplist,tol=1e-5 ):
	"""Is a sequence of maps between a chain complex a chain map?"""
	for i in range(len(chaincomplex1)+1):
		if i==0:
			pass
		elif 0<i<=len(chaincomplex1)+1:
			composition1=np.dot(chaincomplex2[i-1],maplist[i-1])
			composition2=np.dot(maplist[i],chaincomplex1[i-1])
			difference=np.dot(chaincomplex2[i-1],maplist[i-1])-np.dot(maplist[i],chaincomplex1[i-1])
			norm = np.linalg.norm(difference)
			if norm>tol:
			 	#print "No chain map"
			 	return False
	else:
		#print "Chain map!"
		return True

def composeChainMaps( maplist1, maplist2 ):
	"""Compose chain maps"""
	composeChainMaps = []
	for i in range(len(maplist1)): # Assumes number of maps in each map list are equal
		ind_composition = np.dot(maplist1[i],maplist2[i])
	composeChainMaps.append(ind_composition)
	return composeChainMaps

def inducedMap( chaincomplex1, chaincomplex2, maplist, k, tol=1e-5 ):
	"""Compute induced map on homology"""
	h=homology(chaincomplex1,k)
	l=homology(chaincomplex2,k)
	if np.dot(maplist[k],h).shape[1]==0:
		induced_map=np.zeros((l.shape[1],0))
	else:
		induced_map=np.linalg.lstsq(np.dot(maplist[k],h),l)
		induced_map=induced_map[0]
	return induced_map




