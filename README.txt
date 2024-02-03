This directory contains the source code of the project, please execute it one by one in file order after installing the dependencies in requirements.txt.

Since the original CT image data was too large (20Gib), we could not upload it all together. So, we extracted the features after the model went through the extractor and stabiliser and saved them in the Data directory. The Data\Original directory in the root directory holds the raw data that has not been stabilised by the stabiliser. The OtherModules folder in this directory holds the source code related to the stabiliser, resampler and network entry feature extractor.

If the runtime environment is installed correctly, it may take more than two hours for all the code to run, thanks for understanding!