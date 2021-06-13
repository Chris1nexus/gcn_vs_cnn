
import os
import requests
import re
import zipfile

def recursive_visit(curr_path, samples, filename_pattern, replacement_string, counter):
            '''
            Create list of the paths of the (image,mask)pairs, fetched according to the given filename regex pattern
            Args:
              curr_path  (str) current_path 
              samples    list of ( str, str ) pairs
              filename_pattern (regex pattern)
              replacement_string (replace path dependent part with a unique identifier)
              counter (int) count of found images
            '''
            
            for item in os.listdir(curr_path):
                curr_item_path = os.path.join(curr_path, item)
                if os.path.isfile(curr_item_path):
                    filename = item
                    filepath = curr_item_path
                    
                    file_match = filename_pattern.search(filename)
                    # if not none and the three groups sample_type={crop, segmentation_crop}, coord={x[0-9]_y[0-9], png} exist
                    if file_match is not None and len(file_match.groups()) == 3:
                            sample_type, coord, filetype = file_match.groups()
                            if "png" in filetype:
                                if "seg" in sample_type:
                                    
                                    filepath_image = os.path.join(curr_path, replacement_string + coord +".png" )
                                    
                                    if os.path.exists(filepath_image):
                                        samples.append((filepath_image, curr_item_path))
                                        counter += 1
                            
                if os.path.isdir(curr_item_path):
                    curr_dir_path = curr_item_path
                    counter = recursive_visit(curr_dir_path, samples, filename_pattern, replacement_string, counter)
                    
            return counter





class DriveDownloader(object):
  
  def __init__(self):
    pass
  def download_file_from_google_drive(self, id, destination):
      URL = "https://docs.google.com/uc?export=download"

      session = requests.Session()

      response = session.get(URL, params = { 'id' : id }, stream = True)
      token = DriveDownloader.__get_confirm_token__(response)

      if token:
          params = { 'id' : id, 'confirm' : token }
          response = session.get(URL, params = params, stream = True)

      DriveDownloader.__save_response_content__(response, destination)   
     

  def __get_confirm_token__(response):
      for key, value in response.cookies.items():
          if key.startswith('download_warning'):
              return value

      return None

  def __save_response_content__(response, destination):
      CHUNK_SIZE = 32768

      with open(destination, "wb") as f:
          for chunk in response.iter_content(CHUNK_SIZE):
              if chunk: # filter out keep-alive new chunks
                  f.write(chunk)
  def extract_zip(self, source, target_directory):
    with zipfile.ZipFile(source, 'r') as zip_ref:
        zip_ref.extractall(target_directory)





  def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')