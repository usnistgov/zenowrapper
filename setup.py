from setuptools import setup
from Cython.Build import cythonize

extensions = [
    Extension("primes", ["primes.pyx"],
        include_dirs=[...],
        libraries=[...],
        library_dirs=[...]),
    # Everything but primes.pyx is included here.
    Extension("*", ["*.pyx"],
        include_dirs=[...],
        libraries=[...],
        library_dirs=[...]),
]
setup(
    name="My hello app",
    ext_modules=cythonize(extensions),
)