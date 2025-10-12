import numpy as np

# Find (unique) 2D-3D correspondences from 2D-2D correspondences
def Find2D3DCorrespondences(image_name, images, matches, registered_images):
  assert(image_name not in registered_images)

  image_kp_idxs = []
  p3D_idxs = []
  for other_image_name in registered_images:
    other_image = images[other_image_name]
    pair_matches = GetPairMatches(image_name, other_image_name, matches)

    for i in range(pair_matches.shape[0]):
      p3D_idx = other_image.GetPoint3DIdx(pair_matches[i,1])
      if p3D_idx > -1:
        p3D_idxs.append(p3D_idx)
        image_kp_idxs.append(pair_matches[i,0])

  print(f'found {len(p3D_idxs)} points, {np.unique(np.array(p3D_idxs)).shape[0]} unique points')

  # Remove duplicated correspondences
  _, unique_idxs = np.unique(np.array(p3D_idxs), return_index=True)
  image_kp_idxs = np.array(image_kp_idxs)[unique_idxs].tolist()
  p3D_idxs = np.array(p3D_idxs)[unique_idxs].tolist()

  return image_kp_idxs, p3D_idxs


# Make sure we get keypoint matches between the images in the order that we requested
def GetPairMatches(im1, im2, matches):
  if im1 < im2:
    return matches[(im1, im2)]
  else:
    return np.flip(matches[(im2, im1)], 1)


# Update the reconstruction with the new information from a triangulated image
def UpdateReconstructionState(new_points3D, corrs, points3D, images):

  # TODO
  # Add the new points to the set of reconstruction points and add the correspondences to the images.
  # Be careful to update the point indices to the global indices in the `points3D` array.

  M = new_points3D.shape[0]
  if M == 0:
    print("[Update] No new points to add.")
    return points3D, images

  offset = points3D.shape[0]

  # Append new points
  points3D = np.vstack([points3D, new_points3D])

  ### DEBUG ###
  print(f"[Update] added {M} points (global size: {offset} -> {points3D.shape[0]})")
  ### DEBUG ###

  #Add 2D–3D correspondences to each image (convert local -> global indices)
  for im_name, pairs in corrs.items():
    if not pairs:
      continue
    kp_idxs, local_idxs = zip(*pairs)
    kp_idxs = list(kp_idxs)

    global_p3D_idxs = [offset + local_idx for local_idx in local_idxs]

    images[im_name].Add3DCorrs(kp_idxs, global_p3D_idxs)

    ### DEBUG ###
    print(f"[Update] {im_name}: added {len(kp_idxs)} 2D-3D corrs")
    ### DEBUG ###

  return points3D, images