# Makefile for 1D Heat Diffusion Simulation
# Level 0 testbed for Fortran -> Python/JAX migration

FC = gfortran
FFLAGS = -O2 -Wall -Wextra -fcheck=all
LDFLAGS =

# Source files (order matters for dependencies)
SRCS = params_mod.f90 init_mod.f90 simulation_mod.f90 output_mod.f90 heat1d.f90

# Object files
OBJS = $(SRCS:.f90=.o)

# Executable name
TARGET = heat1d

# Default target
all: $(TARGET)

# Link
$(TARGET): $(OBJS)
	$(FC) $(FFLAGS) -o $@ $^ $(LDFLAGS)

# Compile with dependencies
params_mod.o: params_mod.f90
	$(FC) $(FFLAGS) -c $<

init_mod.o: init_mod.f90 params_mod.o
	$(FC) $(FFLAGS) -c $<

simulation_mod.o: simulation_mod.f90 params_mod.o
	$(FC) $(FFLAGS) -c $<

output_mod.o: output_mod.f90 params_mod.o
	$(FC) $(FFLAGS) -c $<

heat1d.o: heat1d.f90 params_mod.o init_mod.o simulation_mod.o output_mod.o
	$(FC) $(FFLAGS) -c $<

# Run simulation
run: $(TARGET)
	mkdir -p output
	./$(TARGET)

# Clean
clean:
	rm -f $(TARGET) *.o *.mod

cleanall: clean
	rm -rf output/

# Phony targets
.PHONY: all run clean cleanall
