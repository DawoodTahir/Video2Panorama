import os
import cv2
import numpy as np

def extractFrames(videoPath, outputFolder, interval, maxFrames):
	# Ensure the output folder exists
	os.makedirs(outputFolder, exist_ok=True)

	cap = cv2.VideoCapture(videoPath)
	fps = cap.get(cv2.CAP_PROP_FPS)
	success, frameCount, extractFrames = True, 0, 0

	while success and extractFrames < maxFrames:
		success, frame = cap.read()
		if not success:
			break
		if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) % int(fps * interval) == 0:
			frameName = f"IMG_{frameCount}.jpg"
			cv2.imwrite(os.path.join(outputFolder, frameName), frame)
			frameCount += 1
			extractFrames += 1

	cap.release()
	print(f"Extracted {frameCount} frames.")


def imageReading(folderPath):
	# Input Images will be stored in this list.
	images = []

	# Checking if the path is a directory
	if os.path.isdir(folderPath):
		# Get all files in the folder and sort them based on numeric value in filenames
		imageNames = os.listdir(folderPath)

		#Extracting numbers from filenames for sorting
		imageNames_split = [(int(os.path.splitext(filename)[0].split('_')[1]), filename)
					for filename in imageNames if filename.lower().endswith('.jpg')]
		imageNames_split.sort(key=lambda x: x[0])
		imageNames_sorted = [item[1] for item in imageNames_split]

		#Reading and resizing images
		for imageName in imageNames_sorted:
			imgPath = os.path.join(folderPath, imageName)
			inputImage = cv2.imread(imgPath)

			if inputImage is not None:
				inputImage = cv2.resize(inputImage, (1080,720))
				images.append(inputImage)
			else:
				print(f"Not able to read image: {imageName}")
				exit(0)
	else:
		print("\nEnter a valid image folder path.")

	if len(images) < 2:
		print("\n Not enough images found. Please provide 2 or more images")
		exit(1)

	return images

def findMatches(baseImage, secImage):
	# Using SIFT to find the keypoints and descriptors of the images.
	sift = cv2.SIFT_create()
	baseImage_kp, baseImage_des = sift.detectAndCompute(cv2.cvtColor(baseImage,cv2.COLOR_BGR2GRAY),None) # kp is for keypoints, des is for descriptors
	secImage_kp, secImage_des = sift.detectAndCompute(cv2.cvtColor(secImage,cv2.COLOR_BGRA2BGR),None)

	# Using Brute Force matcher to find matches
	bf = cv2.BFMatcher()
	initialMatches = bf.knnMatch(baseImage_des,secImage_des,k=2)

	# Applying ratio test and filtering out the good matches
	goodMatches = []
	for m,n in initialMatches:
		if m.distance < 0.75 * n.distance:
			goodMatches.append([m])

	return goodMatches, baseImage_kp, secImage_kp

def findingHomography(matches, baseImage_kp, secImage_kp):
	# If less than 4 matches found, exit code
	if len(matches) < 4:
		print("\nNot enough matches found between the images.")
		exit(0)

	# Storing the coordinates of points corresponding to the matches found in both the images
	baseImage_pts = []
	secImage_pts = []
	for match in matches:
		baseImage_pts.append(baseImage_kp[match[0].queryIdx].pt)
		secImage_pts.append(secImage_kp[match[0].trainIdx].pt)

	# Changing the datatype to "float32" for finding homography
	baseImage_pts = np.float32(baseImage_pts)
	secImage_pts = np.float32(secImage_pts)

	# Finding the homography matrix (transformation matrix)
	(homographyMatrix, status) = cv2.findHomography(secImage_pts, baseImage_pts, cv2.RANSAC, 4.0)

	return homographyMatrix, status

def getNewFrame(homographyMatrix, secImage_shape, baseImage_shape):

	# Reading the size of the image
	(h,w) = secImage_shape

	# Taking the matrix of initial coordinates of the corners of the secondary image
	initialMatrix =np.array([[0, w - 1, w - 1, 0],
							[0, 0, h - 1, h - 1],
							[1, 1, 1, 1]])

	# Finding the final coordinates of the corners of the image after transformation
	finalMatrix = np.dot(homographyMatrix,initialMatrix)
	
	[x, y, c] = finalMatrix
	x = np.divide(x, c)
	y = np.divide(y, c)

	# Finding the dimensions of the stitched image frame and the 'correction' factor
	min_x, max_x = int(round((min(x)))), int(round(max(x)))
	min_y, max_y = int(round((min(y)))), int(round(max(y)))
	
	newWidth = max_x
	newHeight = max_y
	correction = [0, 0]
	if min_x < 0:
		newWidth -= min_x
		correction[0] = abs(min_x)
	if min_y < 0:
		newHeight -= min_y
		correction[1] = abs(min_y)

	# Again correcting New_Width and New_Height
	if newWidth < baseImage_shape[1] + correction[0]:
		newWidth = baseImage_shape[1] + correction[0]
	if newHeight < baseImage_shape[0] + correction[1]:
		newHeight = baseImage_shape[0] + correction[1]

	# Finding the coordinates of the corners of the image if they all were within the frame.
	x = np.add(x, correction[0])
	y = np.add(y, correction[1])
	oldInitialPoints = np.float32([[0, 0],
								   [w - 1, 0],
								   [w - 1, h - 1],
								   [0, h - 1]])
	newFinalPoints = np.float32(np.array([x, y]).transpose())

	# Updating the homography matrix. Done so that now the secondary image completely
	# lies inside the frame
	HomographyMatrix = cv2.getPerspectiveTransform(oldInitialPoints, newFinalPoints)

	return [newHeight, newWidth], correction, HomographyMatrix

def stitchImages(baseImage, secImage):
	# Applying Cylindrical projection on SecImage
	secImage_Cyl, mask_x, mask_y = projection(secImage)

	# Getting SecImage Mask
	secImage_Mask = np.zeros(secImage_Cyl.shape, dtype=np.uint8)
	secImage_Mask[mask_y, mask_x, :] = 255

	# Finding matches between the 2 images and their keypoints
	matches, baseImage_kp, secImage_kp = findMatches(baseImage, secImage_Cyl)

	# Finding homography matrix.
	homographyMatrix, status = findingHomography(matches, baseImage_kp, secImage_kp)

	# Finding size of new frame of stitched images and updating the homography matrix
	newFrameSize, correction, homographyMatrix = getNewFrame(homographyMatrix, secImage_Cyl.shape[:2], baseImage.shape[:2])

	# Finally placing the images upon one another.
	SecImage_Transformed = cv2.warpPerspective(secImage_Cyl, homographyMatrix, (newFrameSize[1], newFrameSize[0]))
	SecImage_Transformed_Mask = cv2.warpPerspective(secImage_Mask, homographyMatrix, (newFrameSize[1], newFrameSize[0]))
	BaseImage_Transformed = np.zeros((newFrameSize[0], newFrameSize[1], 3), dtype=np.uint8)
	BaseImage_Transformed[correction[1]:correction[1]+baseImage.shape[0], correction[0]:correction[0]+baseImage.shape[1]] = baseImage

	StitchedImage = cv2.bitwise_or(SecImage_Transformed, cv2.bitwise_and(BaseImage_Transformed, cv2.bitwise_not(SecImage_Transformed_Mask)))

	return StitchedImage

def convert_xy(x, y):
	global center, f

	xt = (f * np.tan((x - center[0]) / f)) + center[0]
	yt = ((y - center[1]) / np.cos((x - center[0]) / f)) + center[1]

	return xt, yt

def projection(initialImage):
	global w, h, center, f
	h, w = initialImage.shape[:2]
	center = [w // 2, h // 2]
	f = 1250  # 1500 field; 1000 Sun; 1500 Rainier; 1050 Helens

	# Creating a blank transformed image
	transformedImage = np.zeros(initialImage.shape, dtype=np.uint8)

	# Storing all coordinates of the transformed image in 2 arrays (x and y coordinates)
	allCoordinates_of_ti = np.array([np.array([i, j]) for i in range(w) for j in range(h)])
	ti_x = allCoordinates_of_ti[:, 0]
	ti_y = allCoordinates_of_ti[:, 1]

	# Finding corresponding coordinates of the transformed image in the initial image
	ii_x, ii_y = convert_xy(ti_x, ti_y)

	# Rounding off the coordinate values to get exact pixel values (top-left corner)
	ii_tl_x = ii_x.astype(int)
	ii_tl_y = ii_y.astype(int)

	# Finding transformed image points whose corresponding
	# initial image points lies inside the initial image
	goodIndices = (ii_tl_x >= 0) * (ii_tl_x <= (w - 2)) * \
				  (ii_tl_y >= 0) * (ii_tl_y <= (h - 2))

	# Removing all the outside points from everywhere
	ti_x = ti_x[goodIndices]
	ti_y = ti_y[goodIndices]

	ii_x = ii_x[goodIndices]
	ii_y = ii_y[goodIndices]

	ii_tl_x = ii_tl_x[goodIndices]
	ii_tl_y = ii_tl_y[goodIndices]

	# Bi linear interpolation
	dx = ii_x - ii_tl_x
	dy = ii_y - ii_tl_y

	weight_tl = (1.0 - dx) * (1.0 - dy)
	weight_tr = (dx) * (1.0 - dy)
	weight_bl = (1.0 - dx) * (dy)
	weight_br = (dx) * (dy)

	transformedImage[ti_y, ti_x, :] = (weight_tl[:, None] * initialImage[ii_tl_y, ii_tl_x, :]) + \
									  (weight_tr[:, None] * initialImage[ii_tl_y, ii_tl_x + 1, :]) + \
									  (weight_bl[:, None] * initialImage[ii_tl_y + 1, ii_tl_x, :]) + \
									  (weight_br[:, None] * initialImage[ii_tl_y + 1, ii_tl_x + 1, :])

	# Getting x coordinate to remove black region from right and left in the transformed image
	min_x = min(ti_x)

	# Cropping out the black region from both sides (using symmetric)
	transformedImage = transformedImage[:, min_x: -min_x, :]

	return transformedImage, ti_x - min_x, ti_y


videoPath = "data_3.mp4"
outputFolder = "data_3_frames"
frameInterval = 1 # Extract 1 frame per second
maxFrames = 10
extractFrames(videoPath, outputFolder, frameInterval, maxFrames)

# Reading images.
images = imageReading(outputFolder)

baseImage, _, _ = projection(images[0])
for i in range(1, len(images)):

	stitchedImage = stitchImages(baseImage, images[i])
	baseImage = stitchedImage.copy()

cv2.imwrite("StitchedPanorama360.png", baseImage)
cv2.imshow("Stitched Image", baseImage)
cv2.waitKey()
cv2.destroyAllWindows()