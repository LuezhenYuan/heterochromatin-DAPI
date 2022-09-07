from nd2reader import ND2Reader
from skimage import io
import numpy as np
from scipy import ndimage
from skimage import morphology, filters

def read_nd2_3D(filename):
  """
  Open a nd2 image. It should be 
  Args:
    filename: Name of the image file.
  Returns:
    An OpenSlide object representing a whole-slide image.
  """
  try:
    img = ND2Reader(filename)
    if len(img.metadata['channels']) > 0:
      img.bundle_axes = 'zyx'
      img.iter_axes = 'c'
      img_0 = img[0]
  except:
    print("An exception occurred in opening nd2 file.")
  return img_0

def read_tif_3D(filename):
  """
  Open a tif image.
  Args:
    filename: Name of the image file.
  Returns:
    An OpenSlide object representing a whole-slide image.
  """
  try:
    img = io.imread(img_file_name)
  except:
    print("An exception occurred in opening tif file.")
  return img

def shuffle_labels_notcontinuous(labels):
    random_label_dict = np.unique(labels)
    random_label_dict = random_label_dict[random_label_dict!=0]
    random_label_dict = dict(zip(random_label_dict, np.random.permutation(np.arange(1,len(random_label_dict)+1))))
    random_labels = np.zeros_like(labels)
    for i in random_label_dict:
        random_labels[labels==i] = random_label_dict[i]
    return random_labels

def segmentation_nucleus_otsu_3D(img_0,img_pixel_microns,arg_nucleus_diameter_micron=10,img_zstep_size=2):
  nucleus_size_threshold = 4/3*np.pi*(arg_nucleus_diameter_micron/2)**3/img_pixel_microns/img_pixel_microns/img_zstep_size
  local_Otsu_radius = int(arg_nucleus_diameter_micron*1.0/img_pixel_microns/2 + 0.5)
  (img_zlevels,img_width,img_height)=img_0.shape
  
  ## convert to 8 bit
  img_raw_min,img_raw_max =img_0.min(), img_0.max()
  img_sample_0_uint8 = 255*1.0/(img_raw_max-img_raw_min) * (img_0-img_raw_min)
  img_sample_0_uint8 = img_sample_0_uint8.astype(np.uint8)
  
  img_sample_0_uint8_median= np.zeros((img_zlevels,img_height,img_width),dtype=np.uint8)
  img_sample_0_uint8_gaussian= np.zeros((img_zlevels,img_height,img_width),dtype=np.uint8)
  for z in range(img_zlevels):
      img_sample_0_uint8_median[z,:,:] = ndimage.median_filter(img_sample_0_uint8[z,:,:], size=3)
      img_sample_0_uint8_gaussian[z,:,:] = ndimage.gaussian_filter(img_sample_0_uint8_median[z,:,:], sigma=3)
  
  if local_Otsu_radius<3:
      local_Otsu_radius=3
  mask_disk = morphology.disk(local_Otsu_radius).astype(np.bool_)
  local_otsu_binarized_img_0 = np.zeros((img_zlevels,img_height,img_width),dtype=np.bool_)
  for z in range(img_zlevels):
      img_sample_uint8_local_otsu_slice = filters.rank.otsu(img_sample_0_uint8_gaussian[z,:,:],mask_disk)
      local_otsu_binarized_img_0[z,:,:] = img_sample_0_uint8_gaussian[z,:,:] >= img_sample_uint8_local_otsu_slice
  
  val = filters.threshold_otsu(img_sample_0_uint8_gaussian)
  rough_binarized_img_0 = img_sample_0_uint8_gaussian >= val
  
  binarized_img_0_improved = np.logical_and(rough_binarized_img_0, local_otsu_binarized_img_0)
  
  binarized_img_0_improved_fill_z = np.zeros((img_zlevels,img_height,img_width),dtype=np.bool_)
  for z in range(img_zlevels):
      binarized_img_0_improved_fill = ndimage.morphology.binary_fill_holes(binarized_img_0_improved[z,:,:]).astype(np.bool_) # till 2018-08028, cannot be used for 3d filling holes.
      binarized_img_0_improved_fill_z[z,:,:] = binarized_img_0_improved_fill
  
  # filter 
  labeled_img_0, num_label_0 = ndimage.label(binarized_img_0_improved_fill_z)
  cc_area = ndimage.sum(binarized_img_0_improved_fill_z, labeled_img_0, range(0,np.max(labeled_img_0)+1)) # label to the values of the array, sum of the values for certain labels.
  label_img=np.copy(labeled_img_0)
  volume_mask = (cc_area<nucleus_size_threshold/5) | (cc_area>nucleus_size_threshold*5)
  for i in range(len(volume_mask)):
    if volume_mask[i] == True:
      label_img[label_img==i] = 0
  # low intensity region mask
  cc_intensity = ndimage.mean(img_sample_0_uint8_gaussian, labeled_img_0, range(0,np.max(labeled_img_0)+1))
  intensity_mask = cc_intensity<1.0*val
  for i in range(len(volume_mask)):
    if intensity_mask[i] == True:
      label_img[label_img==i] = 0
  # saturation mask
  saturated_area = ndimage.sum(img_sample_0_uint8==255, labeled_img_0, range(0,np.max(labeled_img_0)+1))
  saturation_mask = 1.0*saturated_area/cc_area >0.1
  for i in range(len(volume_mask)):
    if saturation_mask[i] == True:
      label_img[label_img==i] = 0
  label_img_shuffle = shuffle_labels_notcontinuous(label_img)
  return label_img_shuffle

def segmentation_nucleus_otsu_2D(img_0,img_pixel_microns,arg_nucleus_diameter_micron=10):
  nucleus_size_threshold = np.pi*(arg_nucleus_diameter_micron/2)**2/img_pixel_microns/img_pixel_microns
  local_Otsu_radius = int(arg_nucleus_diameter_micron*1.0/img_pixel_microns/2 + 0.5)
  (img_width,img_height)=img_0.shape
  
  ## convert to 8 bit
  img_raw_min,img_raw_max =img_0.min(), img_0.max()
  img_sample_0_uint8 = 255*1.0/(img_raw_max-img_raw_min) * (img_0-img_raw_min)
  img_sample_0_uint8 = img_sample_0_uint8.astype(np.uint8)
  
  img_sample_0_uint8_median= ndimage.median_filter(img_sample_0_uint8, size=3)
  img_sample_0_uint8_gaussian= ndimage.gaussian_filter(img_sample_0_uint8_median, sigma=3)
  
  if local_Otsu_radius<3:
      local_Otsu_radius=3
  mask_disk = morphology.disk(local_Otsu_radius).astype(np.bool_)
  img_sample_uint8_local_otsu_slice = filters.rank.otsu(img_sample_0_uint8_gaussian,mask_disk)
  local_otsu_binarized_img_0 = img_sample_0_uint8_gaussian >= img_sample_uint8_local_otsu_slice
  
  val = filters.threshold_otsu(img_sample_0_uint8_gaussian)
  rough_binarized_img_0 = img_sample_0_uint8_gaussian >= val
  
  binarized_img_0_improved = np.logical_and(rough_binarized_img_0, local_otsu_binarized_img_0)
  
  binarized_img_0_improved_fill = ndimage.morphology.binary_fill_holes(binarized_img_0_improved).astype(np.bool_) # till 2018-08028, cannot be used for 3d filling holes.
  
  # filter 
  labeled_img_0, num_label_0 = ndimage.label(binarized_img_0_improved_fill)
  cc_area = ndimage.sum(binarized_img_0_improved_fill, labeled_img_0, range(0,np.max(labeled_img_0)+1)) # label to the values of the array, sum of the values for certain labels.
  label_img=np.copy(labeled_img_0)
  volume_mask = (cc_area<nucleus_size_threshold/5) | (cc_area>nucleus_size_threshold*5)
  for i in range(len(volume_mask)):
    if volume_mask[i] == True:
      label_img[label_img==i] = 0
  # low intensity region mask
  cc_intensity = ndimage.mean(img_sample_0_uint8_gaussian, labeled_img_0, range(0,np.max(labeled_img_0)+1))
  intensity_mask = cc_intensity<1.0*val
  for i in range(len(volume_mask)):
    if intensity_mask[i] == True:
      label_img[label_img==i] = 0
  # saturation mask
  saturated_area = ndimage.sum(img_sample_0_uint8==255, labeled_img_0, range(0,np.max(labeled_img_0)+1))
  saturation_mask = 1.0*saturated_area/cc_area >0.1
  for i in range(len(volume_mask)):
    if saturation_mask[i] == True:
      label_img[label_img==i] = 0
  label_img_shuffle = shuffle_labels_notcontinuous(label_img)
  return label_img_shuffle