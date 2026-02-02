module simulation_mod
  !---------------------------------------------------------------------------
  ! Core simulation routines for 1D Heat Diffusion
  ! 
  ! Method: Forward Time Central Space (FTCS) explicit scheme
  ! 
  ! u(i,n+1) = u(i,n) + r * [u(i-1,n) - 2*u(i,n) + u(i+1,n)]
  ! where r = alpha * dt / dx²
  !
  ! Stability condition: r <= 0.5 (CFL condition)
  !---------------------------------------------------------------------------
  use params_mod
  implicit none

contains

  subroutine timestep_ftcs(u, u_new)
    !-------------------------------------------------------------------------
    ! Perform one FTCS time step
    !-------------------------------------------------------------------------
    implicit none
    real(8), intent(in)  :: u(nx)
    real(8), intent(out) :: u_new(nx)
    
    real(8) :: r
    integer :: i

    ! Compute stability parameter
    r = alpha * dt / (dx * dx)

    ! Update interior points using FTCS scheme
    do i = 2, nx - 1
      u_new(i) = u(i) + r * (u(i-1) - 2.0d0 * u(i) + u(i+1))
    end do

    ! Enforce Dirichlet boundary conditions
    u_new(1)  = 0.0d0
    u_new(nx) = 0.0d0
  end subroutine timestep_ftcs

  function compute_error(u, u_exact) result(l2_error)
    !-------------------------------------------------------------------------
    ! Compute L2 norm of error between numerical and analytical solution
    !-------------------------------------------------------------------------
    implicit none
    real(8), intent(in) :: u(nx)
    real(8), intent(in) :: u_exact(nx)
    real(8) :: l2_error
    real(8) :: sum_sq
    integer :: i

    sum_sq = 0.0d0
    do i = 1, nx
      sum_sq = sum_sq + (u(i) - u_exact(i))**2
    end do
    l2_error = sqrt(sum_sq / nx)
  end function compute_error

  function compute_max_error(u, u_exact) result(max_error)
    !-------------------------------------------------------------------------
    ! Compute maximum (L-infinity) error
    !-------------------------------------------------------------------------
    implicit none
    real(8), intent(in) :: u(nx)
    real(8), intent(in) :: u_exact(nx)
    real(8) :: max_error
    integer :: i

    max_error = 0.0d0
    do i = 1, nx
      max_error = max(max_error, abs(u(i) - u_exact(i)))
    end do
  end function compute_max_error

  function compute_total_heat(u) result(total)
    !-------------------------------------------------------------------------
    ! Compute total heat content (integral of u over domain)
    ! Uses trapezoidal rule
    !-------------------------------------------------------------------------
    implicit none
    real(8), intent(in) :: u(nx)
    real(8) :: total
    integer :: i

    total = 0.5d0 * (u(1) + u(nx))
    do i = 2, nx - 1
      total = total + u(i)
    end do
    total = total * dx
  end function compute_total_heat

end module simulation_mod
