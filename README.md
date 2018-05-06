# Some codes for treating parafermion chains

A python package for simulating 1D quantum systems using tensor networks and exact diagonalisation. 
Particular focus on parafermionic chains.

## Installation

On a debian based Linix distribution the following commands should work. 

```sudo apt-get install python-pip```

Clone the repository with 

```git clone https://github.com/nmoran/pf_resonances.git```

Install any required python packages listed in the requirements file with

```pip install -r requirements.txt```

Build and install using

```
python setup.py build
python setup.py install 
```

## Using the codes from scripts

The parafermion code can be imported wuth

```import parafermions as pf```

See the scripts folder for examples of some scripts that use the code.

## Using the codes from the command line

After installation the utilities `PFData.py`, `PFPT.py` and `PFFullDiag.py` wil be in the path. 
To use each of these utilities one can access their help using the `--help` switch. 

