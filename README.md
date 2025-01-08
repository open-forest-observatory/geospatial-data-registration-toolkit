# Geospatial Data Registration Toolkit
The goal of this work is to register geospatial data (ex. orthomosaics or digital terrain models) in a simple and robust way. This tool handles the pre-processing work to pre-process the geospatial data into a form that can be used by standard image registration algorithms. The backend algorithm can be swapped out, and we currently have several options from the`opencv` and `sitk` libraries.

# Install
To set up this project you can run the following commands. This assumes `anaconda` and `poetry` are installed.
```
conda create -n GDRT python=3.10 -y
conda activate GDRT
poetry install
```

# Examples
Example data can be found in this [folder](https://ucdavis.box.com/v/GDRT-example-data). The contents should be downloaded and placed in the `data` folder. The two notebooks in `dev` can be run using this data.
