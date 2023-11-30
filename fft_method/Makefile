INC=/usr/include
NPYLIB=lib/npy.o
NPYMOD=lib/m_npy.o
NPYSRC=src/npy.f90
OPT=-Ofast -march=native
RWLIB=lib/rw_diff.o
RWSRC=src/rw_diff.f90
RWMOD=lib/m_rw_diff.mod
FLAGS=-lfftw3 -lfftw3_omp -lm -fopenmp -Wpedantic -Werror
DIRICHLET=prg/dirichlet

MAIN=prg/main

$(MAIN): $(MAIN).f90 $(NPYLIB) $(RWLIB)
	@echo "---> Compiling $(MAIN)"
	gfortran $(MAIN).f90 -o $(MAIN) $(NPYLIB) $(RWLIB) -Ilib/ $(OPT) $(FLAGS)

$(NPYLIB): $(NPYSRC)
	@echo "---> Compiling npy reader/writer module"
	gfortran -c $(NPYSRC) -o $(NPYLIB) $(OPT) -Jlib/

$(RWLIB): $(RWSRC)
	@echo "---> Compiling Richards-Wolf diffraction module"
	gfortran -c $(RWSRC) -o $(RWLIB) $(FLAGS) -I$(INC) $(OPT) -Jlib/

$(DIRICHLET): $(DIRICHLET).f90 $(NPYLIB) $(RWLIB)
	@echo "---> Compiling dirichlet program"
	gfortran $(DIRICHLET).f90 -o $(DIRICHLET) $(RWLIB) $(NPYLIB) -Ilib/ $(OPT) $(FLAGS)

dirichlet: $(DIRICHLET)

all: $(DIRICHLET) $(MAIN) $(NPYLIB) $(RWLIB)

.PHONY: clean
clean:
	rm -rfv *.o *.mod
