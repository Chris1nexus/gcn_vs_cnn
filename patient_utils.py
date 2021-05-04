
from itertools import groupby
from functools import  reduce
from operator import itemgetter

""" Helper classes to group samples of the same patient"""


class Patient(object):
  '''
  Patient that represents all samples of the RCC dataset, to it related.
  It is used for the purpose of standardizing by patient.
  A sample group represents all samples related to a given annotation of a given patient: all samples are grouped under the same group if no annotation subset is given.
  '''

  def __init__(self, patientid, cancer_type, dataset):
    self.patientid = patientid
    self.cancer_type = cancer_type
    self.dataset = dataset
    self.sample_groups = {}

    # used for samples without key 
    self.standard_key = 0
  def add_sample(self, img_path, seg_path, sample_key=None):
    '''
    Add sample to the sample group of the current patient
    A sample key is composed of all items in the path of the given sample, FROM the PATIENT ID(included) 
    till the filename component (EXCLUDED).

    '''
    if sample_key is None:
      sample_key = self.standard_key

    sample_list = self.sample_groups.get(sample_key,None)

    if sample_list is None:
      sample_list = []
      self.sample_groups[sample_key] = sample_list

    sample_list.append((img_path, seg_path) )
  

  @staticmethod
  def __add_all(destination_patient, source_patient):
   
    for sample_item in source_patient.sample_groups.items():
      sample_key, sample_list = sample_item

      merged_sample_list = destination_patient.sample_groups.get(sample_key,None)

      if merged_sample_list is None:
        merged_sample_list = []
        destination_patient.sample_groups[sample_key] = merged_sample_list

      merged_sample_list.extend(sample_list)


  @staticmethod
  def merge_patients(patient1,patient2):
    patientid1 = patient1.patientid
    patientid2 = patient2.patientid

    # in this dataset a patient can only be categorized by a type of cancer, but in real world samples,
    # two different wsi slices from the same patient could show different results (e.g. one with traces of cancer, while the other shows up as healthy) 
    assert patientid1 == patientid2, "Error: patients don't have the same id"
    assert patient1.cancer_type ==  patient2.cancer_type, "Error: two different cancer types for the same patient"
    assert patient1.dataset ==  patient2.dataset, "Error: two different dataset types for the same patient"
    
    merged_patient = Patient(patientid1, patient1.cancer_type, patient1.dataset)

    Patient.__add_all(merged_patient, patient1)
    Patient.__add_all(merged_patient, patient2)

  
    return merged_patient


def find_position(item , document):
  try:
    location = document.index(item)
  except:
    location = -1
  return location  


def map_patients_to_ids(sample):
  '''
  Generate key-value mapping from a given sample, key = patientid, value=Patient object
  This is the map phase needed for the consequent reduce, that groups all samples of a given patients, by merging two 'patient' objects with the same id together.
  Initially, a patient is generated for each sample. Then, these are reduced together through the map-reduce paradigm.
  Args:
      -sample tuple(img_path: str, seg_path: str) a single sample composed by the path of the image and the path of its mask
  Returns:
      - patient tuple (patientid:str,patient:Patient) 
  '''
  img_path, seg_path = sample

  train_str = "Train"
  test_str = "Test"
  train_information_start = find_position(train_str, img_path)
  test_information_start = find_position(test_str, img_path)
  position = max(train_information_start, test_information_start)

  assert position >= 0 , "Information has not been found in text"
  img_path_shortened = img_path[position:]
  seg_path_shortened = seg_path[position:] 

  img_path_data = img_path_shortened.split("/")
  seg_path_data = seg_path_shortened.split("/")

  # train/test
  sample_split_location = img_path_data[0]
  # category 
  sample_label = img_path_data[1]
  # patientid 
  patientid = img_path_data[2]
  
  img_filename = img_path_data[-1]
  seg_filename = seg_path_data[-1] 

  patient = Patient(patientid, sample_label, sample_split_location)
  components_of_the_key = img_path_data[2:-1]  # key is composed by patientid + any of the following folders to avoid conflicts due to equal subfolder names
  sample_key = "/".join( components_of_the_key  )
  
  patient.add_sample(img_path, seg_path,  sample_key)

  return (patientid, patient)


def reduce_bykey(patient1, patient2):
    '''
    Reduce phase to apply after mapping with map_patients_to_ids is done (patients passed to this function have the same id by definition of the MAP REDUCE PARADIGM)
    Args:
      patient1 tuple (patientid:str,patient:Patient) 
      patient2 tuple (patientid:str,patient:Patient)
    Returns:
      merged patient tuple (patientid:str,patient:Patient)

    '''
    pid1, patient1 = patient1
    pid2, patient2 = patient2
    return (pid1, Patient.merge_patients(patient1,patient2))
