import os, sys
driect_file_path = os.path.abspath(__file__)
foward_file_path = os.path.dirname(driect_file_path)
sys.path.append(foward_file_path)

from .ModelType import ModelType
from .DataProvider import DataProvider
from .AutoModel import AutoModel