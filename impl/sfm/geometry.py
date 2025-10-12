import numpy as np

from impl.dlt import BuildProjectionConstraintMatrix
from impl.util import MakeHomogeneous, HNormalize
from impl.sfm.corrs import GetPairMatches
# from impl.opt import ImageResiduals, OptimizeProjectionMatrix

# # Debug
# import matplotlib.pyplot as plt
# from impl.vis import Plot3DPoints, PlotCamera, PlotProjectedPoints


def EstimateEssentialMatrix(K, im1, im2, matches):
  # TODO
  # Normalize coordinates (to points on the normalized image plane)
  # Convert pixel coordinates (u, v) into normalized camera coordinates (x, y, 1).

  # First, get the keypoint coordinates that are part of the matches
  kps1_matched = im1.kps[matches[:,0]]
  kps2_matched = im2.kps[matches[:,1]]

  #convert to homogeneous coordinates (add a 1)
  hom_kps1 = np.hstack((kps1_matched, np.ones((kps1_matched.shape[0], 1)))) # (N,3)
  hom_kps2 = np.hstack((kps2_matched, np.ones((kps2_matched.shape[0], 1)))) # (N,3)

  # Normalize by multiplying with the inverse of the intrinsic matrix K
  normalized_kps1 = (np.linalg.inv(K) @ hom_kps1.T).T  # (N,3)
  normalized_kps2 = (np.linalg.inv(K) @ hom_kps2.T).T  # (N,3)

  #Set last coord to 1 for numeric stability
  normalized_kps1 = normalized_kps1 / normalized_kps1[:,[2]]
  normalized_kps2 = normalized_kps2 / normalized_kps2[:,[2]]

  # Assemble constraint matrix as equation 1
  # For each pair of matched points p1 and p2, create a row
  # for the constraint matrix A such that the row multiplied
  # by the vectorized Essential Matrix E equals zero.
  constraint_matrix = np.zeros((matches.shape[0], 9))
  for i in range(matches.shape[0]):
    # TODO
    # Add the constraints
    # For each match (p1, p2), the constraint is p2^T * E * p1 = 0.
    # This can be rewritten as a linear equation A * vec(E) = 0.
    # The row of A is derivedd from the elements of p1 and p2.

    #compute the kronecker product p2.T ⊗ p1.T
    constraint_matrix[i] = np.kron(normalized_kps2[i], normalized_kps1[i])

  # Solve for the nullspace of the constraint matrix
  _, _, vh = np.linalg.svd(constraint_matrix)
  vectorized_E_hat = vh[-1]

  # TODO
  # Reshape the vectorized matrix to it's proper shape again (from (9,1) to (3,3))
  E_hat = vectorized_E_hat.reshape(3,3, order='F')

  # TODO
  # We need to fulfill the internal constraints of E
  # The first two singular values need to be equal, the third one zero.
  # Since E is up to scale, we can choose the two equal singluar values arbitrarily

  #The estimated matrix E_hat is noisy. We need to find the closest valid
  # Essential Matrix to it. A valid E has singular values of [σ, σ, 0].
  U, S, Vh = np.linalg.svd(E_hat)
  S_clean = np.diag([1,1,0])
  E = U @ S_clean @ Vh

  # This is just a quick test that should tell you if your estimated matrix is not correct
  # It might fail if you estimated E in the other direction (i.e. kp2' * E * kp1)
  # You can adapt it to your assumptions.
  # Vectorized algebraic residual |x2^T E x1| for ALL points
  x1n = normalized_kps1   # (N,3)
  x2n = normalized_kps2   # (N,3)

  # Residual for x2^T E x1
  res21 = np.abs(np.sum(x2n * ((E @ x1n.T).T), axis=1))

  # Also compute the opposite convention; if it's better, transpose E
  res12 = np.abs(np.sum(x1n * ((E @ x2n.T).T), axis=1))
  if np.median(res12) < np.median(res21):
      E = E.T
      res21 = np.abs(np.sum(x2n * ((E @ x1n.T).T), axis=1))

  # Assert for ALL points; 1e-2 is fine for algebraic residual after K^{-1} normalization
  assert np.all(res21 < 1e-2), f"max algebraic residual {res21.max():.3e}"


  return E


def DecomposeEssentialMatrix(E):

  u, s, vh = np.linalg.svd(E)

  # Determine the translation up to sign
  t_hat = u[:,-1]

  W = np.array([
    [0, -1, 0],
    [1, 0, 0],
    [0, 0, 1]
  ])

  # Compute the two possible rotations
  R1 = u @ W @ vh
  R2 = u @ W.transpose() @ vh

  # Make sure the orthogonal matrices are proper rotations (Determinant should be 1)
  if np.linalg.det(R1) < 0:
    R1 *= -1

  if np.linalg.det(R2) < 0:
    R2 *= -1

  # Assemble the four possible solutions
  sols = [
    (R1, t_hat),
    (R2, t_hat),
    (R1, -t_hat),
    (R2, -t_hat)
  ]

  return sols

def TriangulatePoints(K, im1, im2, matches):

  R1, t1 = im1.Pose()
  R2, t2 = im2.Pose()
  P1 = K @ np.append(R1, np.expand_dims(t1, 1), 1)
  P2 = K @ np.append(R2, np.expand_dims(t2, 1), 1)

  # Ignore matches that already have a triangulated point
  new_matches = np.zeros((0, 2), dtype=int)

  num_matches = matches.shape[0]
  for i in range(num_matches):
    p3d_idx1 = im1.GetPoint3DIdx(matches[i, 0])
    p3d_idx2 = im2.GetPoint3DIdx(matches[i, 1])
    if p3d_idx1 == -1 and p3d_idx2 == -1:
      new_matches = np.append(new_matches, matches[[i]], 0)


  num_new_matches = new_matches.shape[0]

  points3D = np.zeros((num_new_matches, 3))

  for i in range(num_new_matches):

    kp1 = im1.kps[new_matches[i, 0], :]
    kp2 = im2.kps[new_matches[i, 1], :]

    # H & Z Sec. 12.2
    A = np.array([
      kp1[0] * P1[2] - P1[0],
      kp1[1] * P1[2] - P1[1],
      kp2[0] * P2[2] - P2[0],
      kp2[1] * P2[2] - P2[1]
    ])

    _, _, vh = np.linalg.svd(A)
    homogeneous_point = vh[-1]
    points3D[i] = homogeneous_point[:-1] / homogeneous_point[-1]


  # We need to keep track of the correspondences between image points and 3D points
  im1_corrs = new_matches[:,0]
  im2_corrs = new_matches[:,1]

  # TODO
  # Filter points behind the cameras by transforming them into each camera space and checking the depth (Z)
  # Make sure to also remove the corresponding rows in `im1_corrs` and `im2_corrs`

  # Filter points behind the first camera
  # Transform points to camera 1's coordinate system
  points3D_cam1 = (R1 @ points3D.T + t1[:, np.newaxis]).T

  #Create a boolean mask for points with positive depth
  mask1 = points3D_cam1[:, 2] > 0

  # Apply the mask
  im1_corrs = im1_corrs[mask1]
  im2_corrs = im2_corrs[mask1]
  points3D = points3D[mask1]

  # Filter points behind the second camera
  # Transform the remaining points to camera 2's coordinate system
  points3D_cam2 = (R2 @ points3D.T + t2[:, np.newaxis]).T

  #create a bool mask for points w/ positive depth
  mask2 = points3D_cam2[:,2] > 0

  #apply the mask
  im1_corrs = im1_corrs[mask2]
  im2_corrs = im2_corrs[mask2]
  points3D = points3D[mask2]

  return points3D, im1_corrs, im2_corrs

def EstimateImagePose(points2D, points3D, K):

  # TODO
  # We use points in the normalized image plane.
  # This removes the 'K' factor from the projection matrix.
  # We don't normalize the 3D points here to keep the code simpler.

  #assert that we have the same number of points in 2d and 3d:
  assert points2D.shape[0] == points3D.shape[0]

  #get number of 2d points
  N = points2D.shape[0]
  ones = np.ones((N,1)) #

  pts2_h = np.hstack([points2D, ones]) # (N,3) #extend to 3rd dim
  pts2_n = (np.linalg.inv(K) @ pts2_h.T).T # normalize by K
  normalized_points2D = pts2_n[:, :2] / pts2_n[:, [2]]   # drop homogeneous scale

  constraint_matrix = BuildProjectionConstraintMatrix(normalized_points2D, points3D)

  ### DEBUG ###
  rank = np.linalg.matrix_rank(constraint_matrix)
  print(f"[Pose] N={N}, A.shape={constraint_matrix.shape}, rank={rank}")
  ### DEBUG ###

  # We don't use optimization here since we would need to make sure to only optimize on the se(3) manifold
  # (the manifold of proper 3D poses). This is a bit too complicated right now.
  # Just DLT should give good enough results for this dataset.

  # Solve for the nullspace
  _, _, vh = np.linalg.svd(constraint_matrix)
  P_vec = vh[-1,:]
  P = np.reshape(P_vec, (3, 4), order='C')

  # Make sure we have a proper rotation
  u, s, vh = np.linalg.svd(P[:,:3])
  R = u @ vh

  if np.linalg.det(R) < 0:
    R *= -1

  # Camera center from P's nullspace, then t = -R * C
  _, _, vh = np.linalg.svd(P)
  C = np.copy(vh[-1,:])

  t = -R @ (C[:3] / C[3])

  ### DEBUG ###

  # Report rotation quality
  detR = np.linalg.det(R)
  ortho_err = np.linalg.norm(R.T @ R - np.eye(3))
  print(f"[Pose] det(R)={detR:.6f}, ||R^T R - I||={ortho_err:.2e}")

  # Reprojection error in pixels
  Pfull = K @ np.hstack([R, t.reshape(3,1)])
  X_h = np.hstack([points3D, np.ones((points3D.shape[0],1))])
  proj_h = (Pfull @ X_h.T).T
  proj = proj_h[:, :2] / proj_h[:, [2]]
  err = np.linalg.norm(proj - points2D, axis=1)
  print(f"[Pose] reproj px: mean={err.mean():.3f}, med={np.median(err):.3f}, p95={np.percentile(err,95):.3f}, max={err.max():.3f}")

  # In-front check
  X_cam = (R @ points3D.T + t[:, None]).T
  infront_ratio = np.mean(X_cam[:,2] > 0)
  print(f"[Pose] in-front ratio={infront_ratio:.3f}")
  assert infront_ratio > 0.6, f"[Pose] too few points in front: {infront_ratio:.3f}"

  ### DEBUG ###

  return R, t

def TriangulateImage(K, image_name, images, registered_images, matches):

  # TODO
  # Loop over all registered images and triangulate new points with the new image.
  # Make sure to keep track of all new 2D-3D correspondences, also for the registered images

  #get new image
  image = images[image_name]

  #Collect 3D points in chunks, then stack at the end
  points3D_chunks = []  # list of (Pi, 3) arrays

  # You can save the correspondences for each image in a dict and refer to the `local` new point indices here.
  # Afterwards you just add the index offset before adding the correspondences to the images.
  corrs = {} # maps image_name -> list of (kp_idx_in_that_image, local_p3D_idx)

  running_offset = 0 #the count of points collected so far in this call

  ### DEBUG ###
  print(f"[TriImg] new={image_name}, against {len(registered_images)} registered")
  ### DEBUG ###

  for reg_name in registered_images:
     # Get matches in the right order for (new image, registered image)
    pair = GetPairMatches(image_name, reg_name, matches)

    ### DEBUG ###
    print(f"[TriImg] pair {image_name} ↔ {reg_name}: matches={pair.shape[0]}", end="")
    ### DEBUG ###

    #triangulate the points for this pair
    pts, im_corrs, im_reg_corrs = TriangulatePoints(K, image, images[reg_name], pair)

    #if no points yet, continue
    if pts.shape[0] == 0:
      continue

    # Local indices for this batch in the final stacked points array
    local_idxs = np.arange(pts.shape[0]) + running_offset
    running_offset += pts.shape[0]

    #store 3d chunk
    points3D_chunks.append(pts)

    # Record correspondences for the new image
    corrs.setdefault(image_name, [])
    corrs[image_name].extend(list(zip(im_corrs.tolist(), local_idxs.tolist())))

    # Record correspondences for the paired registered image
    corrs.setdefault(reg_name, [])
    corrs[reg_name].extend(list(zip(im_reg_corrs.tolist(), local_idxs.tolist())))

    # Stack all newly created points
  if len(points3D_chunks) == 0:
    points3D = np.zeros((0, 3))
  else:
    points3D = np.vstack(points3D_chunks)

  ### DEBUG ###
  total_new = points3D.shape[0]
  print(f"[TriImg] total new 3D points for {image_name}: {total_new}")
  got_new_corrs = len(corrs.get(image_name, []))
  print(f"[TriImg] corrs[{image_name}] entries={got_new_corrs}")
  if total_new > 0:
    # Each new point must appear exactly once for the new image
    assert got_new_corrs == total_new, f"[TriImg] expected {total_new} corrs for {image_name}, got {got_new_corrs}"
  # Summaries per registered image
  for rn in registered_images:
    if rn in corrs:
      print(f"[TriImg] corrs[{rn}] entries={len(corrs[rn])}")

  ### DEBUG ###

  return points3D, corrs

