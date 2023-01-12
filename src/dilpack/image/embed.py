# Implentation of http://Doi.org/10.54623/fue.fcij.6.2.2
# Implementation of stegonography using QR factorization
# Author: Joseph Emmanuel Dayo

import numpy as np
from numpy import asarray
from PIL import Image
from dilpack.linalg.qr_decompose import computeQR

def __image_to_float(image, grayscale=False, invert_secret_image=False):
  image_array = asarray(image)

  if image_array.ndim == 3:
    r_matrix = image_array[:,:,0:1].astype(np.uint32)
    g_matrix = image_array[:,:,1:2].astype(np.uint32)
    b_matrix = image_array[:,:,2:3].astype(np.uint32)
    m, n, _ = np.shape(r_matrix)
  else:
    grayscale = True
    r_matrix = image_array.astype(np.uint32)
    m, n = np.shape(r_matrix)

  # add noise to prevent norms from being zero
  decomposed_image = np.random.uniform(low=0.1, high=0.2, size=[m, n]).astype(np.float64)
  for j in range(m):
    for k in range(n):
      if grayscale:
        if invert_secret_image:
          decomposed_image[j, k] += 255 - r_matrix[j, k]
        else:
          decomposed_image[j, k] += r_matrix[j, k]
      else:
        decomposed_image[j, k] += (b_matrix[j, k] << 16) | (g_matrix[j, k] << 8) | (r_matrix[j, k])
  return decomposed_image

def __float_to_image(image_array, grayscale=False):
  Qsi_discrete = image_array.astype(np.uint32)
  m, n = np.shape(image_array)
  new_image = np.zeros([m, n, 3], dtype=np.uint8)
  for i in range(m):
    for j in range(n):
      if grayscale:
        new_image[i, j, 2] = Qsi_discrete[i, j] & 0xff
        new_image[i, j, 1] = Qsi_discrete[i, j] & 0xff
        new_image[i, j, 0] = Qsi_discrete[i, j] & 0xff
      else:
        new_image[i, j, 2] = (Qsi_discrete[i, j] >> 16) & 0xff
        new_image[i, j, 1] = (Qsi_discrete[i, j] >> 8) & 0xff
        new_image[i, j, 0] = Qsi_discrete[i, j] & 0xff
  return Image.fromarray(new_image)




def embed(cover_image, secret_image, alpha=0.1, invert_secret_image = False):
  """
  Embeds a secret image into a cover_image, acting as a sort of watermark

  :param cover_image: Image object from Pillow to be used as cover image
  :param secret_image: Image object from Pillow to be used as secret image. Should be monochrome/grayscale for it to work properly
  :param alpha: Optional format override.  If omitted, the
      format to use is determined from the filename extension.
      If a file object was used instead of a filename, this
      parameter should always be used.
  :param invert_secret_image: If True, secret image is inverted, black on white, white on black
  :returns: (Stego-Image in PIL format, Q secret image, R cover image, Image raw array)
  """
  ImageC = __image_to_float(cover_image)
  ImageS = __image_to_float(secret_image, grayscale=True, invert_secret_image=invert_secret_image)
  Qc, Rc = computeQR(ImageC)
  Qs, Rs = computeQR(ImageS)

  m ,n  = np.shape(Qc)
  new_image = np.zeros([n, m])
  Rn = Rc + (alpha * Rs)
  si = np.matmul(Qc, Rn)
  new_image = __float_to_image(si)
  return new_image, Qs, Rc, si

def extract(stego_image, Qs, Rc, alpha = 0.1):
  """
  Extracts a secret image from a stego image

  :param Qs: Orthogonal matrix for the secret image (From the embed)
  :param Rc: Upper Triangular (R) matrix from the Cover Image
  :param alpha: The same alpha used in the embed
  :returns: Image
  """
  img_decomp = __image_to_float(stego_image)
  _, Rsi = computeQR(img_decomp)
  Rnew = (Rsi - Rc) / alpha
  embedded_image = np.matmul(Qs, Rnew)
  return __float_to_image(embedded_image, grayscale=True)