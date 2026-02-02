program heat1d
  !---------------------------------------------------------------------------
  ! 1D Heat Diffusion Simulation
  !
  ! Solves: ∂u/∂t = α * ∂²u/∂x²
  ! Method: Forward Time Central Space (FTCS) explicit scheme
  !
  ! Domain: x ∈ [0, L]
  ! Boundary conditions: u(0,t) = u(L,t) = 0 (Dirichlet)
  ! Initial condition: u(x,0) = sin(π*x/L)
  ! Analytical solution: u(x,t) = sin(π*x/L) * exp(-α*(π/L)²*t)
  !
  ! Based on classic finite difference methods for PDEs.
  ! Designed as Level 0 testbed for Fortran → Python/JAX migration.
  !---------------------------------------------------------------------------
  use params_mod
  use init_mod
  use simulation_mod
  use output_mod

  implicit none

  ! Arrays
  real(8) :: x(nx)           ! Spatial grid
  real(8) :: u(nx)           ! Temperature at current time
  real(8) :: u_new(nx)       ! Temperature at next time
  real(8) :: u_exact(nx)     ! Analytical solution

  ! Time stepping variables
  real(8) :: t               ! Current time
  integer :: step            ! Time step counter

  ! Error tracking
  real(8) :: l2_error, max_error, total_heat

  !---------------------------------------------------------------------------
  ! Initialization
  !---------------------------------------------------------------------------
  
  call print_params()
  
  ! Initialize grid and temperature field
  call init_grid(x)
  call init_temperature(x, u)
  
  ! Initial state
  t = 0.0d0
  step = 0
  
  ! Compute initial errors
  u_exact = analytical_solution(x, t)
  l2_error = compute_error(u, u_exact)
  max_error = compute_max_error(u, u_exact)
  total_heat = compute_total_heat(u)
  
  ! Write initial state
  call write_solution(x, u, u_exact, step, t)
  call write_history(step, t, l2_error, max_error, total_heat)
  
  write(*,'(A)') ""
  write(*,'(A)') "Starting time integration..."
  write(*,'(A)') "Step      Time          L2 Error      Max Error"
  write(*,'(A)') "------------------------------------------------"
  write(*,'(I6,F12.6,ES14.6,ES14.6)') step, t, l2_error, max_error

  !---------------------------------------------------------------------------
  ! Main time loop
  !---------------------------------------------------------------------------
  
  do while (t < t_end)
    ! Advance one time step
    call timestep_ftcs(u, u_new)
    
    ! Update solution
    u = u_new
    t = t + dt
    step = step + 1
    
    ! Compute errors
    u_exact = analytical_solution(x, t)
    l2_error = compute_error(u, u_exact)
    max_error = compute_max_error(u, u_exact)
    total_heat = compute_total_heat(u)
    
    ! Output
    if (mod(step, output_freq) == 0) then
      call write_solution(x, u, u_exact, step, t)
      call write_history(step, t, l2_error, max_error, total_heat)
      write(*,'(I6,F12.6,ES14.6,ES14.6)') step, t, l2_error, max_error
    end if
  end do

  !---------------------------------------------------------------------------
  ! Final output
  !---------------------------------------------------------------------------
  
  ! Write final state
  call write_solution(x, u, u_exact, step, t)
  call write_history(step, t, l2_error, max_error, total_heat)
  call write_final_report(step, t, l2_error, max_error)
  
  write(*,'(A)') "------------------------------------------------"
  write(*,'(A)') "Simulation completed successfully."

end program heat1d
