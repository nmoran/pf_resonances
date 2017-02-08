from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import Cython.Compiler.Options
import numpy
import sysconfig
import sys
import os
import subprocess

Cython.Compiler.Options.annotate = True

def runcommand(cmd):
    process = subprocess.Popen(cmd.split(), shell=False, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, universal_newlines=True)
    c = process.communicate()

    if process.returncode != 0:
        raise Exception("Something went wrong whilst running the command: %s" % cmd)

    return c[0]

def whichmpi():
    # Figure out which MPI environment this is
    import re
    mpiv = runcommand('mpirun -V')

    if re.search('Intel', mpiv):
        return 'intelmpi'
    elif re.search('Open MPI', mpiv):
        return 'openmpi'

    warnings.warn('Unknown MPI environment.')
    return None


def whichscalapack():
    # Figure out which Scalapack to use
    if 'MKLROOT' in os.environ:
        return 'intelmkl'
    else:
        return 'netlib'

# Set the MPI version
mpiversion = whichmpi()

# Set the Scalapack version
scalapackversion = whichscalapack()

# Set to use OMP
use_omp = False


## Find the MPI arguments required for building the modules.
if mpiversion == 'intelmpi':
    # Fetch command line, convert to a list, and remove the first item (the command).
    intelargs = runcommand('mpicc -show').split()[1:]
    mpilinkargs = intelargs
    mpicompileargs = intelargs
elif mpiversion == 'openmpi':
    # Fetch the arguments for linking and compiling.
    mpilinkargs = runcommand('mpicc -showme:link').split()
    mpicompileargs = runcommand('mpicc -showme:compile').split()
else:
    raise Exception("MPI library unsupported. Please modify setup.py manually.")


## Find the Scalapack library arguments required for building the modules.
if scalapackversion == 'intelmkl':
    # Set library includes (taking into account which MPI library we are using)."
    scl_lib = ['mkl_scalapack_ilp64', 'mkl_rt', 'mkl_blacs_'+mpiversion+'_ilp64', 'iomp5', 'pthread']
    scl_libdir = [os.environ['MKLROOT']+'/lib/intel64' if 'MKLROOT' in os.environ else '']
elif scalapackversion == 'netlib':
    scl_lib = ['scalapack-openmpi', 'gfortran']
    scl_libdir = [ os.path.dirname(runcommand('gfortran -print-file-name=libgfortran.a')) ]
else:
    raise Exception("Scalapack distribution unsupported. Please modify setup.py manually.")

omp_args = ['-fopenmp'] if use_omp else []


setup(
    name='parafermions',
    version='0.01',
    description='Some utilities for manipulating parafermion systems.',
    author='Niall Moran',
    author_email='niall.moran@gmail.com',
    license = 'GPLv3',
    cmdclass = {'build_ext': build_ext},
    packages = ['parafermions'],
    package_dir = {'.' : '.'},
    scripts = ['parafermions/PFFullDiag.py', 'parafermions/PFPT.py', 'parafermions/PFData.py'],
    ext_modules = [Extension("parafermions.scalapack_wrapper",
                             ["parafermions/scalapack_wrapper.pyx"],
                             include_dirs=[numpy.get_include()],
                             library_dirs=scl_libdir, libraries=scl_lib,
                             extra_compile_args=mpicompileargs,
                             extra_link_args=mpilinkargs),
                   Extension("parafermions.ParafermionUtilsCython",
                             ["parafermions/ParafermionUtilsCython.pyx"],
                             include_dirs=[numpy.get_include()]
                             )]
)
