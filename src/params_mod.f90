module params_mod
  !---------------------------------------------------------------------------
  ! Parameters for 1D Heat Diffusion Simulation
  ! 
  ! Physical problem: ∂u/∂t = α * ∂²u/∂x²
  ! Domain: x ∈ [0, L]
  ! Boundary conditions: u(0,t) = u(L,t) = 0 (Dirichlet)
  ! Initial condition: u(x,0) = sin(π*x/L)
  ! 
  ! Analytical solution: u(x,t) = sin(π*x/L) * exp(-α*(π/L)²*t)
  !---------------------------------------------------------------------------
  implicit none

  ! Grid parameters
  integer, parameter :: nx = 101          ! Number of spatial points
  real(8), parameter :: L = 1.0d0         ! Domain length [m]
  real(8), parameter :: dx = L / (nx - 1) ! Spatial step size

  ! Physical parameters  
  real(8), parameter :: alpha = 0.01d0    ! Thermal diffusivity [m²/s]

  ! Time parameters
  real(8), parameter :: t_end = 1.0d0     ! End time [s]
  real(8), parameter :: cfl = 0.4d0       ! CFL number (stability: cfl <= 0.5)
  
  ! Derived time step (stability condition: dt <= dx²/(2*alpha))
  real(8), parameter :: dt = cfl * dx * dx / alpha

  ! Output parameters
  integer, parameter :: output_freq = 100 ! Output every N steps
  character(len=*), parameter :: output_dir = "output/"

  ! Mathematical constants
  real(8), parameter :: pi = 3.14159265358979323846d0

contains

  subroutine print_params()
    implicit none
    
    write(*,'(A)') "============================================"
    write(*,'(A)') "1D Heat Diffusion Simulation Parameters"
    write(*,'(A)') "============================================"
    write(*,'(A,I6)')     "  Grid points (nx):     ", nx
    write(*,'(A,F10.6)')  "  Domain length (L):    ", L
    write(*,'(A,ES12.4)') "  Grid spacing (dx):    ", dx
    write(*,'(A,ES12.4)') "  Diffusivity (alpha):  ", alpha
    write(*,'(A,F10.6)')  "  End time (t_end):     ", t_end
    write(*,'(A,ES12.4)') "  Time step (dt):       ", dt
    write(*,'(A,F10.6)')  "  CFL number:           ", cfl
    write(*,'(A,I6)')     "  Total time steps:     ", int(t_end / dt)
    write(*,'(A)') "============================================"
  end subroutine print_params

end module params_mod
