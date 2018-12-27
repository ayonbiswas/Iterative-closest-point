import numpy as np  #version 1.12.1
import cv2          #openCV version 3.3.0
import os
"""
flann_sift takes two images and uses FLANN with sift with nearest neighbour = 2 to detect mating points

output : lists of matched points in cloud 1 and cloud 2, distance between the matched points

"""

def flann_sift(img1,img2):

    ls_pt1 = []
    ls_pt2 = []
    dist = []
    sift = cv2.xfeatures2d.SIFT_create()


    FLANN_INDEX_KDTREE = 1
    flann_params = dict(algorithm=FLANN_INDEX_KDTREE , trees=5)

    matcher = cv2.FlannBasedMatcher(flann_params , {})

    kpts1 , descs1 = sift.detectAndCompute(img1 , None)


    kpts2 , descs2 = sift.detectAndCompute(img2 , None)

    matches = matcher.knnMatch(descs1 , descs2 , 2)
    matchesMask = [ [ 0 , 0 ] for i in range(len(matches)) ]
    for i , (m1 , m2) in enumerate(matches):
        if m1.distance < 0.7 * m2.distance:
            matchesMask[ i ] = [ 1 , 0 ]

            pt1 = kpts1[ m1.queryIdx ].pt
            pt2 = kpts2[ m1.trainIdx ].pt

            ls_pt1.append(pt1)
            ls_pt2.append(pt2)
            dist.append(m1.distance)

    return [ls_pt1,ls_pt2,dist]

"""
_3d_coordinates takes input as the depth image the list of matched coordinates

output : x,y,z coordinates
"""


def _3d_coordinates(depth_image,coordinates):
    cood = []

    for u,v in coordinates:
        Z = depth_image[int(v),int(u)]
        X = u
        Y = v
        cood.append([X,Y,Z])
    return cood

"""
weights() produce weights by inverting the distances between the points

"""
def weights(dist):
    weights_lst = []
    eps = 0.000001
    for i in xrange(len(dist)):
        w = 1/abs(dist[i]+eps)
        weights_lst.append(w)

    return weights_lst

"""
gaussian() uses gaussian metric to produce weights

"""
def gaussian(dist):

    sigma_2 = 30
    for i in xrange(len(dist)):

        d = -1*(np.power(dist[i],2))/(2*np.power(sigma_2,2))

        dist[i] = d

    weights_lst = np.exp(dist)

    return weights_lst


"""
transform() takes as input the point set 1, point set 2, and the corresponding weights

output : rotation matrix R and translation vector T
"""
def transform(P,Q,W):
    assert P.shape == Q.shape

    dim , n= P.shape

    P_mean = np.average(P,axis  =1,weights= np.diag(W))
    Q_mean = np.average(Q,axis=1,weights=np.diag(W))
    P_mean = P_mean.reshape(dim,1)
    Q_mean = Q_mean.reshape(dim,1)
    P_mean_mat = np.repeat(P_mean,n, axis=1)
    Q_mean_mat = np.repeat(Q_mean,n,axis=1)
    X = np.subtract(P,P_mean_mat)
    Y = np.subtract(Q,Q_mean_mat)

    H = np.dot(np.dot(X,W),Y.T)
    U , S , Vt = np.linalg.svd(H)
    R = np.dot(Vt.T , U.T)

    if np.linalg.det(R) < 0:
        Vt[ n - 1 , : ] *= -1
        R = np.dot(Vt.T , U.T)

    T  = Q_mean - np.dot(R, P_mean)

    return R, T

"""
distance() computes the euclidean norm between two point sets x and y
"""

def distance(x , y):

    x = np.asarray(x).flatten()
    y = np.asarray(y).flatten()
    return np.sqrt(np.sum(np.power((x - y) , 2)))

"""
icp() computes the transformation between the current source and nearest destination points, updates the source point set and repeats the
transformation until the the change in euclidean error between the source and reference is <0.001

output: the final rotation and translation matrix
"""
def icp(A, B ,W,  max_iterations=25, tolerance=0.001):
    A = np.asarray(A).reshape(-1 , 3)
    B = np.asarray(B).reshape(-1 , 3)
    W = np.diag(W)

    assert A.shape == B.shape

    src = np.copy(A.T)

    dst = np.copy(B.T)
    dim , n = src.shape

    prev_error = 0
    it = 0
    for i in range(max_iterations):

        it = i

        r,t = transform(src, dst, W)

        t = np.repeat(t,n,axis=1)






        src = np.dot(r, src)+t

        mean_error = distance(src,dst)

        if np.abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error

    r,t = transform(A.T, src,W)

    return r,t, it+1

"""
visualise() takes in the source image, its depth image, rotation and translation matrix and the image number
it reconstructs the transformed image and displays and saves it at a particular directory
"""
def visualise(img, depth , r, t, i):
    width , length = img.shape


    temp = np.empty((3, width*length) )

    for u in xrange(width):
        for v in xrange(length):

            Z = depth[u,v]
            X = v
            Y = u

            temp[:,v+u*length] = np.array([X,Y,Z])

    y = np.dot(r,temp)+np.repeat(t,length*width,axis = 1)

    _2d_img = y[0:2,:]

    img_intensity = np.zeros((width,length),dtype=np.uint8)

    for u in xrange(width):
        for v in xrange(length):
            x,y = _2d_img[:,v+u*length]

            if (x>=0 and x<length and y>=0 and y<width):

                img_intensity[int(y),int(x)] = img[u,v]

    cv2.imshow("projected",img_intensity)

    cv2.imwrite("D:\\RGBD_dataset\\transformed{}.jpg".format(i), img_intensity)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

"""
main() contructs the both image and depth image vectors and call the above routines to perform icp
"""


def main():
    rgb_path = "D:\\rgb_test"
    depth_path = "D:\\depth_test"
    rgb_img = []
    depth_img = []

    for img_files in os.listdir(rgb_path):

        imge = cv2.imread(os.path.join(rgb_path,img_files))
        img = cv2.cvtColor(imge , cv2.COLOR_RGB2GRAY)

        rgb_img.append(np.array(img))

    for depth_files in os.listdir(depth_path):
        d_imge = cv2.imread(os.path.join(depth_path,depth_files))
        d_img = cv2.cvtColor(d_imge , cv2.COLOR_RGB2GRAY)

        depth_img.append(np.array(d_img))


    for i in xrange(len(rgb_img)-1):
        [match_img1, match_img2,distance_12] = flann_sift(rgb_img[i],rgb_img[i+1])
        cood_src= _3d_coordinates(depth_img[i],match_img1)
        cood_dst = _3d_coordinates(depth_img[i+1],match_img2)

        w = gaussian(distance_12)

        r,t,it = icp(cood_src,cood_dst,w)
        print r
        print t

        visualise(rgb_img[ 0 ] , depth_img[ 0 ] ,  r , t, i)



main()



