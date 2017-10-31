# MATH 496/696 Spring 2016 Project 2
#
# Copyright (c) 2016 by Michael Robinson <michaelr@american.edu>
# Permission is granted to copy and modify for educational use, provided that this notice is retained
# Permission is NOT granted for all other uses -- please contact the author for details

# Load the necessary modules
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches

# Remove the key when you're done writing!
import pr2_key

# This function might be useful!
def ksublists(lst,n,sublist=[]): #Inputs vertices, outputs ASC
    """Iterate over all ordered n-sublists of a list lst"""
    if n==0:
        yield sublist
    else:
        for idx in range(len(lst)):
           item=lst[idx]
           for tmp in ksublists(lst[idx+1:],n-1,sublist+[item]):
              yield tmp

# Copyright (c) 2016 by Brian DiZio <brian.a.dizio@gmail.com>
# Permission is NOT granted for all other uses -- please contact the author for details

def dist(x,y):
    """Compute distance between two points"""
    dist = math.sqrt(np.dot((x-y).T,(x-y)))
    return dist
    # return pr2_key.dist(x,y)

def kfaces(simplex,k):
    """Compute all k-dimensional faces of a simplex"""
    for i in ksublists(simplex,k):
        return [i]          
    return pr2_key.kfaces(simplex,k)

def filtering(kfaces):
    output=[]
    for i in kfaces:
        found=False
        for j in output:
            if set(i).issubset(set(j)) and set(j).issubset(set(i)):
                found=True
                break
        if not found:
            output.append(i)
    return output

def vietorisRips(pts,diameter,maxdim=None):
    """Construct a Vietoris-Rips complex over a point cloud"""
    vertices=[]
    onefaces=[]
    twofaces=[]
    threefaces=[]
    for i in range(len(pts)):
        vertices.append([i])
        for j in range(len(pts)):
            if dist(pts[i],pts[j])<=diameter:
                if i!=j:
                    onefaces.append([i,j]) 
            for k in range(len(pts)):    
                if dist(pts[i],pts[j])<=diameter and dist(pts[i],pts[k])<=diameter and dist(pts[j],pts[k])<=diameter:
                    if i!=j and i!=k and j!=k:
                        twofaces.append([i,j,k])
                for l in range(len(pts)):
                    if dist(pts[i],pts[j])<=diameter:
                        if dist(pts[i],pts[k])<=diameter: 
                            if dist(pts[i],pts[l])<=diameter:
                                if dist(pts[j],pts[k])<=diameter:
                                    if dist(pts[j],pts[l])<=diameter:
                                        if dist(pts[k],pts[l])<=diameter:
                                            if i!=j and i!=k and j!=k:
                                                threefaces.append([i,j,k,l])
    vertices=filtering(vertices)
    onefaces=filtering(onefaces)
    twofaces=filtering(twofaces)
    threefaces=filtering(threefaces)
    vietorisRips = vertices + onefaces + twofaces + threefaces
    return vietorisRips

#    return pr2_key.vietorisRips(pts,diameter,maxdim)

def plot_complex(locations,complex,color=None):
    """Plot an abstract simplicial complex given locations of nodes"""
    xsvertices=[] # Plot vertices variable
    ysvertices=[] # Plot vertices variable
    xsonestotal=[] # Plot one-faces variable
    ysonestotal=[] # Plot one-faces variable
    onefacesbucket=[] # Plot one-faces variable
    twofacesbucket=[] # Plot two-faces variable
    pointstwos=[] # Plot two-faces variable
    codes = [Path.MOVETO, # Plot two-faces variable
            Path.LINETO,
            Path.LINETO,
            Path.CLOSEPOLY,]
    for i in complex:
        if len(i) == 1: 
            index=i[0]
            xsvertices.append(np.asscalar(locations[index][0]))
            ysvertices.append(np.asscalar(locations[index][1]))
        elif len(i)==2: onefacesbucket.append(i) # accumulate all one-faces
        elif len(i)==3: twofacesbucket.append(i) # accumulate all two-faces
        else: pass
    for j in onefacesbucket:
        index1=j[0]
        index2=j[1]
        xpoint1=np.asscalar(locations[index1][0])
        ypoint1=np.asscalar(locations[index1][1])
        xpoint2=np.asscalar(locations[index2][0])
        ypoint2=np.asscalar(locations[index2][1])
        xsones=[xpoint1,xpoint2]
        ysones=[ypoint1,ypoint2]
        xsonestotal.append(xsones)
        ysonestotal.append(ysones) # Separate x and y points 
    for l in twofacesbucket:
        index1=l[0]
        index2=l[1]
        index3=l[2]
        point1=locations[index1]
        point2=locations[index2]
        point3=locations[index3]
        pointend=[0., 0.]
        allpoints=[point1,point2,point3,pointend]
        pointstwos.append(allpoints) # Get x and y for vertices of two-faces
    fig = plt.figure()
    plt.plot(xsvertices,ysvertices,"ro") # Plot vertices
    plt.hold(True)
    for k in range(len(xsonestotal)):       
        plt.plot(xsonestotal[k],ysonestotal[k],'g--') # Plot one faces
        plt.hold(True)
    for m in range(len(pointstwos)):
        path = Path(pointstwos[m], codes) # Plot two faces
        ax = fig.add_subplot(111)
        patch = patches.PathPatch(path, facecolor='orange', lw=0)
        ax.add_patch(patch)

def ksimplices(complex,k):
    """Extract all k-simplices from a simplicial complex"""
    ksimplices = []
    for i in complex:
        if len(i) == k+1:
            ksimplices.append(i)
    return ksimplices

def zerolistmaker(n):
    listofzeros = [0] * n
    return listofzeros

def boundary(complex,k):
    """Compute the k-boundary matrix for a simplicial complex"""
    if k==0:
        matrix=[]
        for i in ksimplices(complex,0):
            matrix.append([])
        boundary = np.array(matrix).T
    else:
        kplusonefaces=ksimplices(complex,k)
        kfaces=ksimplices(complex,k-1)
        boundary=[]
        for i in range(len(kfaces)):
            rowi=zerolistmaker(len(kplusonefaces)) #WLOG use the 0th element
            boundary.append(rowi)
        for i in range(len(kplusonefaces)): # For each kface      # Fill out row by row
            for j in range(len(kfaces)): # for each k-1 face, or 
                checking=kplusonefaces[i][:] # decouple the copy and the original
                for k in kfaces[j]: # for each element in the first kface
                    for l in kplusonefaces[i]: # for each element in the kplusoneface
                        if k == l:
                            checking[checking.index(l)]="Deleted" # If they match, replace it with 0
                    numnondeleted=0
                    for m in checking:
                        if m!="Deleted":
                            numnondeleted+=1
                    indextochange=0
                    if numnondeleted==1:
                        for g in range(len(checking)):
                            if checking[g] != "Deleted":
                                indextochange=g
                        if indextochange%2==0:
                            boundary[j][i]=1
                        else:
                            boundary[j][i]=-1
                    elif numnondeleted>1:
                        boundary[j][i]=0
                    # if kfaces[j].index(k)==len(checking)-1:
                    #     checking=kplusonefaces[i][:] # reset checking conditionally
        if not boundary:
            boundary=np.zeros((0,0))
        boundary=np.array(boundary)
    return boundary 
    
def simplicialChainComplex(complex,ks):
    """Construct the simplicial chain complex from an abstract simplicial complex"""
    simplicialChainComplex = []
    for i in ks: # you will have as many boundary maps as you do degrees
        simplicialChainComplex.append(boundary(complex,i))
    return simplicialChainComplex

