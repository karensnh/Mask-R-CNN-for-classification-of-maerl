"""
Mask R-CNN
Run training and implementation of maskrnn on maerl images.
"""
import glob
import os
import re
from osgeo import gdal
import numpy as np
#import ogr
from osgeo import ogr
import sys
import logging
import random

ROOT_DIR = os.path.abspath('/content/mask_rcnn')
sys.path.append(ROOT_DIR)
from mrcnn import utils
from mrcnn.config import Config



class MaerlShapesConfig(Config):
  '''Configuration for training on maerl imagery.
  Derives from the base Config class and overrides
  values specific to the this dataset.
  '''
  # Give the configuration a recognizable name
  NAME = 'maerl'

  #IMAGES_PER_GPU = 2
  IMAGES_PER_GPU = 1

  #GPU_COUNT = 1
  GPU_COUNT = 2
  
  # Number of classes including background
  NUM_CLASSES = 1 + 1  # Just maerl
  
  IMAGE_RESIZE_MODE = "pad64"
  IMAGE_MAX_DIM = 1024
  IMAGE_MIN_DIM = 1024
  
      # Number of training steps per epoch
    # This doesn't need to match the size of the training set. Tensorboard
    # updates are saved at the end of each epoch, so setting this to a
    # smaller number means getting more frequent TensorBoard updates.
    # Validation stats are also calculated at each epoch end and they
    # might take a while, so don't set this too small to avoid spending
    # a lot of time on validation stats.
  STEPS_PER_EPOCH = 5000 # default is 1000, altered to X5 for augmentation 500#1000

    # Number of validation steps to run at the end of every training epoch.
    # A bigger number improves accuracy of validation stats, but slows
    # down the training.

  
class InferConfig(Config):
  '''Configuration for inference.
  Derives from the base Config class and overrides
  values specific to the this dataset.
  '''
  # Give the configuration a recognizable name
  NAME = 'maerlinfer'

  #IMAGES_PER_GPU = 2
  IMAGES_PER_GPU = 1

  GPU_COUNT = 1

  NUM_CLASSES = 1 + 1  # Just maerl
  
  IMAGE_RESIZE_MODE = "none"#"square" #"crop" #"square"

  IMAGE_MAX_DIM = 3648 #
  FPN_CLASSIF_FC_LAYERS_SIZE =256


  USE_MINI_MASK = False
  


    # Non-maximum suppression threshold for detection
  DETECTION_NMS_THRESHOLD =0.7# as for nucleus example 0.1 #0.3
  
def scale_to_8bit(pix):
  scaled = np.zeros(pix.shape, dtype=float)
  for b in range(len(pix)):
      scaled[b] = pix[b]/pix[b].max()
  
  return (scaled*255).astype('uint8')

def load_as_features(geojson_fn):
  src = ogr.Open(geojson_fn)
  return src

def has_zero_instances(geojson_fn):
  feats = load_as_features(geojson_fn)
  layer = feats.GetLayer()
  if layer.GetFeatureCount() == 0:
    return True
  return False

class MaerlShapes(utils.Dataset):
  def load_maerl(self, dataset_dir, skip_empty=False):
           
    if not os.path.exists(dataset_dir):
      logging.error('Path {} does not exist'.format(dataset_dir))
      return  
    
    image_dir = 'images'
    json_dir = 'geometries'
    print(image_dir)
    print(json_dir)
    print(dataset_dir)
    print(os.path.join(dataset_dir,image_dir))
    #image_files = glob.glob(os.path.join(dataset_dir, image_dir, '*.JPG'))
    #image_files = glob.glob(os.path.join(dataset_dir, image_dir, '*.tif'))+glob.glob(os.path.join(dataset_dir, image_dir, '*.jpg'))
    image_files = glob.glob(os.path.join(dataset_dir, image_dir, '*.tif'))
    #image_files = glob.glob(os.path.join(dataset_dir, image_dir, '*.TIF'))
    #print(image_files1)
    #image_files1 = glob.glob(os.path.join(dataset_dir, image_dir, '*.jpg'))
    #print(image_files1)
    print(image_files)
    
    image_ids = [int(re.findall(r'\d+', img)[0]) for img in image_files]
    print(image_ids)
    # image_ids = [int(re.findall(r'\d+', img)[2]) for img in image_files]
    # json = glob.glob(os.path.join(dataset_dir, json_dir, '*.geojson'))
    #json = glob.glob(os.path.join(dataset_dir, json_dir, '*.json'))
    #print(json[0])
    # json_ids = [int(re.findall(r'\d+', js)[2]) for js in json]
    #json_ids = [int(re.findall(r'\d+', js)[0]) for js in json]
    #print(json_ids[0])
    
    if os.path.exists(os.path.join(dataset_dir, json_dir)):      
      json = glob.glob(os.path.join(dataset_dir, json_dir, '*.json'))
      json_ids = [int(re.findall(r'\d+', js)[0]) for js in json]
      
      for i, id_ in enumerate(image_ids):
        src = gdal.Open(image_files[i])
        if has_zero_instances(json[json_ids.index(id_)]):
          if skip_empty:
            continue # zero instance image

        self.add_image(
          "maerl",
          image_id=id_,
          path=image_files[i],
          width=src.RasterXSize,
          height=src.RasterYSize,
          polygons=json[json_ids.index(id_)],
          geotransform=src.GetGeoTransform()
        )
    else:
      for i, id_ in enumerate(image_ids):
        src = gdal.Open(image_files[i])
        self.add_image(
          "maerl",
          image_id=id_,
          path=image_files[i],
          width=src.RasterXSize,
          height=src.RasterYSize,
          geotransform=src.GetGeoTransform()
        )
    self.add_class('maerl', 1, 'maerl')

    
  def load_image(self, image_id):
    img = gdal.Open(self.image_info[image_id]['path'])
    byte_img = scale_to_8bit(img.ReadAsArray())

    return byte_img.transpose(1, 2, 0)
  
  #from balloon
  def image_reference(self, image_id):
      """Return the id of the image."""
      info = self.image_info[image_id]
      if info["source"] == "maerl":
          return info["id"]
      else:
          super(MaerlShapes, self).image_reference(image_id)
          
  def split_train_test(self, proportion_train=.7):
    '''Returns a train and test dataset based
    on the proportion split'''

    data = self.image_info.copy()
    print(len(data))
    print(type(data))
    print(data[0])
    print(data[1])
    print(data)
    #return(print(data))
    
    random.seed(4) # so that it returns the same training/validation set for each restart of runtime 
    random.shuffle(data)
    #return(print(data))

    n = int(len(data) * proportion_train)
    #return(print(n))
    #return(print(len(data)))
    train = data[:n]
    test = data[n:]

    train_data = MaerlShapes()
    train_data.image_info = train
    train_data.class_info = self.class_info
    train_data.prepare()

    test_data = MaerlShapes()
    test_data.image_info = test
    test_data.class_info = self.class_info
    test_data.prepare()

    return train_data, test_data
  
            
  def load_image(self, image_id):
    img = gdal.Open(self.image_info[image_id]['path'])
    byte_img = scale_to_8bit(img.ReadAsArray())

    return byte_img.transpose(1, 2, 0)
  
  def load_mask(self, image_id):
    '''Load instance masks for the given image.
      
      Returns:
        masks: A bool ndarray of binary masks with
          shape [h, w, instance count]
        class_ids: a 1d ndarray of class IDs of the
          instance masks.
    '''
    image_info = self.image_info[image_id]
    instances = []
    if 'polygons' in image_info:
      feats = load_as_features(image_info['polygons'])
      lyr = feats.GetLayer()

      # Rasterize each feature as a new instance
      driver = gdal.GetDriverByName('MEM')
      for feat in lyr:
        # Each feature is a new layer
        outdriver = ogr.GetDriverByName('MEMORY')
        out = outdriver.CreateDataSource('memData')

        #tmp = outdriver.Open('memData', 1)
        out_layer = out.CreateLayer('layer', geom_type=ogr.wkbPolygon)
        out_layer.CreateFeature(feat)

        mem_raster = driver.Create(
            '',
            image_info['width'],
            image_info['height'],
            gdal.GDT_Byte
        )
        mem_raster.SetGeoTransform(image_info['geotransform'])
        gdal.RasterizeLayer(mem_raster, [1], out_layer, burn_values=[1])

        instances.append(mem_raster.GetRasterBand(1).ReadAsArray())
    
    if len(instances) !=0:
      mask = np.stack(instances, axis=2).astype(np.bool)

      # All the same class
      classes = np.ones(mask.shape[-1], dtype=np.int32)
      
      return mask, classes

    else:
      # Call super class to return an empty mask
      return super(MaerlShapes, self).load_mask(image_id)
