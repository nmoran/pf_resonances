from distutils.core import setup
from distutils.extension import Extension
from disttest import test
from Cython.Distutils import build_ext
import Cython.Compiler.Options
import numpy
import sysconfig
import sys
import os
import subprocess

Cython.Compiler.Options.annotate = True

def runcommand(cmd):
    try:
        process = subprocess.Popen(cmd.split(), shell=False, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, universal_newlines=True)
    except OSError as ioerr:
        return None
    c = process.communicate()

    if process.returncode != 0:
        raise Exception("Something went wrong whilst running the command: %s" % cmd)

    return c[0]

def whichmpi():
    # Figure out which MPI environment this is
    import re
    mpiv = runcommand('mpirun -V')
    if runcommand('mpicc -v') is None:
        return None

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
Extensions = [Extension("parafermions.ParafermionUtilsCython",
                       ["parafermions/ParafermionUtilsCython.pyx"],
                       include_dirs=[numpy.get_include()]
                       )]
mpiversion = whichmpi()

# Set the Scalapack version
scalapackversion = whichscalapack()

# Set to use OMP
use_omp = False
omp_args = ['-fopenmp'] if use_omp else []
has_mpi, has_scalapack = True, True


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
    print("MPI not found. Don't worry, almost everything will work without it.")
    has_mpi = False


## Find the Scalapack library arguments required for building the modules.
if scalapackversion == 'intelmkl':
    # Set library includes (taking into account which MPI library we are using)."
    scl_lib = ['mkl_scalapack_ilp64', 'mkl_rt', 'mkl_blacs_'+mpiversion+'_ilp64', 'iomp5', 'pthread']
    scl_libdir = [os.environ['MKLROOT']+'/lib/intel64' if 'MKLROOT' in os.environ else '']
elif scalapackversion == 'netlib':
    scl_lib = ['scalapack-openmpi', 'gfortran']
    if runcommand('gfortran -v') is None:
        has_scalapack = False
    else:
        scl_libdir = [os.path.dirname(runcommand('gfortran -print-file-name=libgfortran.a')) ]
else:
    print("Scalapack not found. Don't worry, almost everything will work without it.")
    has_scalapack = False

if has_mpi and has_scalapack:
    Extensions.append(Extension("parafermions.scalapack_wrapper",
                                 ["parafermions/scalapack_wrapper.pyx"],
                                 include_dirs=[numpy.get_include()],
                                 library_dirs=scl_libdir, libraries=scl_lib,
                                 extra_compile_args=mpicompileargs,
                                 extra_link_args=mpilinkargs))


setup(
    name='parafermions',
    version='0.01',
    description='Some utilities for manipulating parafermion systems.',
    author='Niall Moran',
    author_email='niall.moran@gmail.com',
    license = 'GPLv3',
    cmdclass = {'build_ext': build_ext, 'test': test},
    packages = ['parafermions'],
    package_dir = {'.' : '.'},
    scripts = ['parafermions/PFFullDiag.py', 'parafermions/PFPT.py', 'parafermions/PFData.py'],
    ext_modules= Extensions,
    options={
      'test':
        {
          'test_dir': ['parafermions/tests']
        }
      }
)
