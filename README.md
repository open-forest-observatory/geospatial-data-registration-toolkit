# Geospatial Data Registration Toolkit
The goal of this work is to register geospatial data (ex. orthomosaics or digital terrain models). This tool relies on registration algorithms implemented in `opencv` and `sitk`.

# Install
To set up this project you can run the following commands. This assumes `anaconda` and `poetry` are installed.
```
conda create -n GDRT python=3.10 -y
conda activate GDRT
poetry install
```

# Data
Example data can be found in this [folder](https://ucdavis.box.com/v/GDRT-example-data). The contents should be downloaded and placed in the `data` folder.