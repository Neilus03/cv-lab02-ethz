import numpy as np
from matplotlib import pyplot as plt
import random

np.random.seed(0)
random.seed(0)

def least_square(x,y):

	# TODO
	# return the least-squares solution
	# you can use np.linalg.lstsq

	A = np.vstack([x, np.ones_like(x)]).T #[x,1]
	(k,b), *_ = np.linalg.lstsq(A,y,rcond=None)

	return k, b #k=slope, b=intercept

def num_inlier(x,y,k,b,n_samples,thres_dist):
	# TODO
	# compute the number of inliers and a mask that denotes the indices of inliers

	# Distance from each point to the line y = kx + b:
	# |k*x - y + b| / sqrt(k^2 + 1)

	denom = np.sqrt(k**2 +1)
	d = np.abs(k*x - y + b) / denom

	mask = d < thres_dist
	num = int(np.count_nonzero(mask))

	return num, mask

def ransac(x,y,iter,n_samples,thres_dist,num_subset):
	# RANSAC for y = kx + b
	# TODO
	# ransac
	k_ransac = None
	b_ransac = None
	inlier_mask = np.zeros(n_samples, dtype=bool)
	best_inliers = 0


	#minimal subset size (2 points for a line)
	m = max(2, int(num_subset))

	for _ in range(iter):
		#get random minimal subset
		idx = np.random.choice(n_samples, size=m, replace=False)
		#provisional fit on the subset with least squares
		A = np.c_[x[idx], np.ones(m)]
		k_hat, b_hat = np.linalg.lstsq(A, y[idx], rcond=None)[0]

		#get score by counting inliers:
		num, mask = num_inlier(x, y, float(k_hat), float(b_hat), n_samples, thres_dist)

		#keep the best so far:
		if num > best_inliers:
			best_inliers = num
			inlier_mask = mask
			k_ransac = float(k_hat)
			b_ransac = float(b_hat)



	if best_inliers > 0:
		A = np.c_[x[inlier_mask], np.ones(best_inliers)]
		k_ref, b_ref = np.linalg.lstsq(A, y[inlier_mask], rcond=None)[0]
		k_ransac = float(k_ref)
		b_ransac = float(b_ref)
	else:
		A = np.c_[x, np.ones(n_samples)]
		k_all, b_all = np.linalg.lstsq(A, y, rcond=None)[0]
		k_ransac = float(k_all)
		b_ransac = float(b_all)
		inlier_mask = np.ones(n_samples, dtype=bool)
        
	return k_ransac, b_ransac, inlier_mask

def main():
	iter = 300
	thres_dist = 1
	n_samples = 500
	n_outliers = 50
	k_gt = 1
	b_gt = 10
	num_subset = 5
	x_gt = np.linspace(-10,10,n_samples)
	print(x_gt.shape)
	y_gt = k_gt*x_gt+b_gt
	# add noise
	x_noisy = x_gt+np.random.random(x_gt.shape)-0.5
	y_noisy = y_gt+np.random.random(y_gt.shape)-0.5
	# add outlier
	x_noisy[:n_outliers] = 8 + 10 * (np.random.random(n_outliers)-0.5)
	y_noisy[:n_outliers] = 1 + 2 * (np.random.random(n_outliers)-0.5)

	# least square
	k_ls, b_ls = least_square(x_noisy, y_noisy)

	# ransac
	k_ransac, b_ransac, inlier_mask = ransac(x_noisy, y_noisy, iter, n_samples, thres_dist, num_subset)
	outlier_mask = np.logical_not(inlier_mask)

	print("Estimated coefficients (true, linear regression, RANSAC):")
	print(k_gt, b_gt, k_ls, b_ls, k_ransac, b_ransac)

	line_x = np.arange(x_noisy.min(), x_noisy.max())
	line_y_ls = k_ls*line_x+b_ls
	line_y_ransac = k_ransac*line_x+b_ransac

	plt.scatter(
	    x_noisy[inlier_mask], y_noisy[inlier_mask], color="yellowgreen", marker=".", label="Inliers"
	)
	plt.scatter(
	    x_noisy[outlier_mask], y_noisy[outlier_mask], color="gold", marker=".", label="Outliers"
	)
	plt.plot(line_x, line_y_ls, color="navy", linewidth=2, label="Linear regressor")
	plt.plot(
	    line_x,
	    line_y_ransac,
	    color="cornflowerblue",
	    linewidth=2,
	    label="RANSAC regressor",
	)
	plt.legend()
	plt.xlabel("Input")
	plt.ylabel("Response")
	plt.show()

if __name__ == '__main__':
	main()