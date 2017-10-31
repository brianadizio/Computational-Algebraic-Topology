# MATH 496/696 Spring 2016 Project 3
#
# Copyright (c) 2016 by Michael Robinson <michaelr@american.edu>
# Permission is granted to copy and modify for educational use, provided that this notice is retained
# Permission is NOT granted for all other uses -- please contact the author for details

# Load the necessary modules
import math
import itertools
import numpy as np

# Remove the key when you're done writing!
import pr3_key

def kernel(A, tol=1e-5):
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

def extendBasis(vectors,tol=1e-5):
    """Return a list of vectors that completes a given list into a basis"""
    subbasis = [v for v in pr3_key.oneStepBarcode(np.eye(vectors[0].shape[0]),vectors,tol) if v != None]
    return np.hsplit(np.hstack(subbasis+[cokernel(np.hstack(subbasis))]),vectors[0].shape[0])

class Bar:
    def __init__(self,vector,startindex=0):
        self.vectors=[None for i in range(startindex)]+[vector]

    def isActive(self):
        return self.vectors[-1] is not None

    def currentVector(self):
        return self.vectors[-1]

    def lifeSpan(self):
        b=0
        d=0
        alive=False
        for i in range(len(self.vectors)):
            if self.vectors == None:
                b += 1
                if alive:
                    d=i
                alive=False
            else:
                alive=True
            
        return (b,d)

def permutationSign(list1,list2):
    """Compute the sign of a permutation between two lists"""
    if len(list1) != len(list2):
        return 0
    sgn = 1
    for lst in itertools.permutations(list1):
        if not [x for x in zip(lst,list2) if x[0] != x[1]]:
            return sgn
        else:
            sgn *= -1
    return 0

# Copyright (c) 2016 by Brian DiZio <brian.a.dizio@gmail.com>
# Permission is NOT granted for all other uses -- please contact the author for details

class SimplicialMap:
    def __init__(self,asc1,asc2,simpmap): # Keep this constructor!
        self.asc1=asc1
        self.asc2=asc2
        self.simpmap=simpmap

    def simplicialChainMap(self,k):
        """Transform a simplicial map into a chain map
        Inputs: asc1, asc2 = two abstract simplicial complexes
                simpmap = list of pairs (source vertex,destination vertex)
                k = degree for which chain map is to be computed
        Returns: the matrix of the chain map
        Assumes: 
        1. the simplicial complexes list all faces without duplicates
        2. the simplicial map has no duplicates, and defines a function
        """
        # Going from acs1 to asc2!!!!!
        asc1kfaces=ksimplices(self.asc1,k)
        asc2kfaces=ksimplices(self.asc2,k)
        asc1kfacestransformed=applyingsimpmap(asc1kfaces,self.simpmap)

        kthchainmap=[]
        for j in range(len(asc2kfaces)): # as many cols as kfaces in ASC2
            kthchainmap.append([0]*len(asc1kfaces))
        for k in range(len(asc1kfacestransformed)): # Compute values column by column
            for l in range(len(asc2kfaces)):
                if set(asc1kfacestransformed[k])==set(asc2kfaces[l]):
                    kthchainmap[l][k]=permutationSign(asc1kfacestransformed[k],asc2kfaces[l])
        kthchainmap=np.array(kthchainmap)
        return kthchainmap

        # # NOTE: Remove BOTH of the following lines when writing this function!
        # sm=pr3_key.SimplicialMap(self.asc1,self.asc2,self.simpmap)
        # return sm.simplicialChainMap(k)
    
def applyingsimpmap(kfaces,simpmap):
    alreadychanged=[]
    for i in range(len(simpmap)):
        for j in range(len(kfaces)):
            for k in range(len(kfaces[j])):
                if [j,k] not in alreadychanged:
                    if kfaces[j][k]==simpmap[i][0]:
                        alreadychanged.append([j,k])
                        kfaces[j][k]=simpmap[i][1]
    return kfaces

def ksimplices(complex,k):
    """Extract all k-simplices from a simplicial complex"""
    ksimplices = []
    for i in complex:
        if len(i) == k+1:
            ksimplices.append(i)
    return ksimplices

def islinearindependent(matrix,tol=1e-5):
    col=np.shape(matrix)[1]
    rank=np.linalg.matrix_rank(matrix,tol)
    if rank<col:
        return False
    else:
        return True

def oneStepBarcode(oneMap,basis,tol=1e-5):
    """Compute the barcode diagram for a one-step persistence module.
    Input consists of a numpy array specifying the map in the persistence module and a list of vectors forming the basis of the domain.
    Output consists of list of vectors in the same order as the basis, but any linear dependencies are marked as Nones
    """
    answerlist=[]
    for i in basis:  # for every column in the kernel
        output = np.dot(oneMap,i)
        if np.linalg.norm(output)<tol:
            answerlist.append(None)
        if np.linalg.norm(output)>=tol and len(answerlist)>0:
            zeros=np.zeros((output.shape[0],0))
            for i in answerlist:
                matrixtocheck=np.concatenate((zeros,i),axis=1)
                matrixtocheck=np.concatenate((matrixtocheck,output),axis=1)
            if islinearindependent(matrixtocheck)==True:
                answerlist.append(output)
            if islinearindependent(matrixtocheck)==False:
                answerlist.append(None)    
        if np.linalg.norm(output)>=tol and len(answerlist)==0:
            answerlist.append(output)
    return answerlist

def computeBarcode(matrices,tol=1e-5):
    """Compute the barcode diagram for an arbitrary persistence module"""
    
    # Generate basis as an nxn identity matrix, then split it up
    basis=np.eye(matrices[0].shape[1],matrices[0].shape[1])
    
    allbars=[]
    if matrices[0].shape[1]==0:
        for i in range(matrices[1].shape[1]):
            x=Bar(None,0)
            allbars.append(x)
    else:
        degzerobasisvectors=np.split(basis,matrices[0].shape[1],axis=1)
        for i in range(matrices[0].shape[1]):
            x=Bar(degzerobasisvectors[i],0)
            allbars.append(x)
    
    makinganewbasis=[]
    activebars=[]
    for j in range(len(matrices)):
        if j==0:

            if matrices[j].shape[1]==0:
                basis=np.eye(matrices[1].shape[1],matrices[1].shape[1])
                makinganewbasis=np.split(basis,matrices[1].shape[1],axis=1)
            # if the first matrix has 0 columns, then skip it
            # and making a new basis should be computed for the next matrix

            else:
                # Run onestepBarcode and append the new vectors to existing bars
                firstpass=oneStepBarcode(matrices[j],degzerobasisvectors)
                for k in range(len(firstpass)):
                    allbars[k].vectors.append(firstpass[k])
                
                # Check active bars, take their vectors, run extendbasis, make a new basis
                for h in allbars:
                    if h.isActive()==True:
                        activebars.append(h)
                for m in activebars:
                    makinganewbasis.append(m.currentVector())
                makinganewbasis=extendBasis(makinganewbasis)

                livevectors=[]
                for f in activebars:
                    livevectors.append(f.currentVector())
                for g in makinganewbasis:
                    if (any((g == x).all() for x in livevectors))==False:
                        activebars.append(Bar(g,j+1))
                        allbars.append(Bar(g,j+1))

        if j>0:
            allothers=oneStepBarcode(matrices[j],makinganewbasis)
            # Run onestepBarcode and append the new vectors to existing bars
            for k in range(len(allothers)):
                allbars[k].vectors.append(allothers[k])

            # Clear old new basis
            del makinganewbasis[:]
            # Clear old activebars
            del activebars[:]
            # Check active bars, take their vectors, run extendbasis, make a new basis
            for h in allbars:
                if h.isActive()==True:
                    activebars.append(h)
            for m in activebars:
                makinganewbasis.append(m.currentVector())
            makinganewbasis=extendBasis(makinganewbasis)
    
            # Create new bars for new basis vectors
            livevectors=[]
            for f in activebars:
                livevectors.append(f.currentVector())
            for g in makinganewbasis:
                if (any((g == x).all() for x in livevectors))==False:
                    activebars.append(Bar(g,j+1))
                    allbars.append(Bar(g,j+1))
            # Clear livevectors for next iteration
            del livevectors[:] 
    return allbars
    # return pr3_key.computeBarcode(matrices)

def filteredComplexToMaps(filteredASC):
    """Compute the simplicial maps for a filtered simplicial complex
    Inputs: filteredASC = a list of pairs: first entry is an integer listing birth time, second entry is the simplex
    Returns: a list of SimplicialMap objects"""
    
    simplicialMaps=[]
    birthtimes=[]
    for i in filteredASC:
        if i[0] not in birthtimes:
            birthtimes.append(i[0])
    birthtimes=sorted(birthtimes)
    for i in birthtimes[:-1]:

        simpmap=[]
        ASC1=[g[1] for g in filteredASC if g[0]<=i]
        ASC2=[g[1] for g in filteredASC if g[0]<=i+1]
        for j in ASC1:
            for k in j:
                if (k,k) not in simpmap:
                    simpmap.append((k,k))
        simplicialMaps.append(SimplicialMap(ASC1,ASC2,simpmap))

    # print "mine"
    # for i in range(len(simplicialMaps)-1):
    #     print "asc1 : %d" %i
    #     print simplicialMaps[i].asc1
    #     print "asc2 : %d" %i
    #     print simplicialMaps[i].asc2
    #     print "simpmap : %d" %i
    #     print simplicialMaps[i].simpmap
    # print ""
    # print "his"
    # print ""
    # for i in range(len(pr3_key.filteredComplexToMaps(filteredASC))):
    #     print "asc1 : %d" %i
    #     print pr3_key.filteredComplexToMaps(filteredASC)[i].asc1
    #     print "asc2 : %d" %i
    #     print pr3_key.filteredComplexToMaps(filteredASC)[i].asc2
    #     print "simpmap : %d" %i
    #     print pr3_key.filteredComplexToMaps(filteredASC)[i].simpmap
    
    return simplicialMaps[:-1]
    
    #return pr3_key.filteredComplexToMaps(filteredASC)

def filteredComplexToBarcode(filteredASC,k,tol=1e-5):
    """Compute the simplicial maps for a filtered simplicial complex
    Inputs: filteredASC = a list of pairs: first entry is an integer listing birth time, second entry is the simplex
    Returns: a list of Bar objects"""
    simplicialMaps=filteredComplexToMaps(filteredASC) #A list of sm objects
    matrices=[]
    for i in range(len(simplicialMaps)):
        # print simplicialMaps[i].asc1
        # print simplicialMaps[i].asc2
        # print simplicialMaps[i].simpmap
        # asc1kfaces=ksimplices(simplicialMaps[i].asc1,k)
        # asc2kfaces=ksimplices(simplicialMaps[i].asc2,k)
        # asc1kfacestransformed=applyingsimpmap(asc1kfaces,simplicialMaps[i].simpmap)
        # print asc1kfacestransformed
        # print asc2kfaces
        # print ""
        matrices.append(simplicialMaps[i].simplicialChainMap(k))
    barcode=computeBarcode(matrices)
    return barcode

