module init_mod
  !---------------------------------------------------------------------------
  ! Initial conditions and grid setup for 1D Heat Diffusion
  !---------------------------------------------------------------------------
  use params_mod
  implicit none

contains

  subroutine init_grid(x)
    !-------------------------------------------------------------------------
    ! Initialize spatial grid
    !-------------------------------------------------------------------------
    implicit none
    real(8), intent(out) :: x(nx)
    integer :: i

    do i = 1, nx
      x(i) = (i - 1) * dx
    end do
  end subroutine init_grid

  subroutine init_temperature(x, u)
    !-------------------------------------------------------------------------
    ! Set initial temperature distribution
    ! Initial condition: u(x,0) = sin(π*x/L)
    !-------------------------------------------------------------------------
    implicit none
    real(8), intent(in)  :: x(nx)
    real(8), intent(out) :: u(nx)
    integer :: i

    do i = 1, nx
      u(i) = sin(pi * x(i) / L)
    end do

    ! Enforce boundary conditions
    u(1)  = 0.0d0
    u(nx) = 0.0d0
  end subroutine init_temperature

  function analytical_solution(x, t) result(u_exact)
    !-------------------------------------------------------------------------
    ! Compute analytical solution at time t
    ! u(x,t) = sin(π*x/L) * exp(-α*(π/L)²*t)
    !-------------------------------------------------------------------------
    implicit none
    real(8), intent(in) :: x(nx)
    real(8), intent(in) :: t
    real(8) :: u_exact(nx)
    real(8) :: decay_rate
    integer :: i

    decay_rate = alpha * (pi / L)**2

    do i = 1, nx
      u_exact(i) = sin(pi * x(i) / L) * exp(-decay_rate * t)
    end do
  end function analytical_solution

end module init_mod
