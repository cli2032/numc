from distutils.core import setup, Extension
import sysconfig

def main():
    CFLAGS = ['-g', '-Wall', '-std=c99', '-fopenmp', '-mavx', '-mfma', '-pthread', '-O3']
    LDFLAGS = ['-fopenmp']
    # Use the setup function we imported and set up the modules.
    # You may find this reference helpful: https://docs.python.org/3.6/extending/building.html
    # TODO: YOUR CODE HERE
    setup(name = "numc",
    	  version = "1.0",
    	  description = "Python interface for numc",
    	  author = "Calvin Li",
    	  author_email = "cli2032@berkeley.edu",
    	  ext_modules = [Extension("numc", ["numc.c", "matrix.c"], 
    	  						   extra_compile_args = CFLAGS,
    	  						   extra_link_args = LDFLAGS)])

if __name__ == "__main__":
    main()
