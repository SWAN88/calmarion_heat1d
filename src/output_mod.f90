module output_mod
  !---------------------------------------------------------------------------
  ! Output routines for 1D Heat Diffusion
  !---------------------------------------------------------------------------
  use params_mod
  implicit none

contains

  subroutine write_solution(x, u, u_exact, step, t)
    !-------------------------------------------------------------------------
    ! Write solution to CSV file
    !-------------------------------------------------------------------------
    implicit none
    real(8), intent(in) :: x(nx)
    real(8), intent(in) :: u(nx)
    real(8), intent(in) :: u_exact(nx)
    integer, intent(in) :: step
    real(8), intent(in) :: t
    
    character(len=256) :: filename
    integer :: i, unit_num
    
    unit_num = 20

    write(filename, '(A,A,I6.6,A)') trim(output_dir), "solution_", step, ".csv"
    
    open(unit=unit_num, file=trim(filename), status='replace')
    
    ! Write header
    write(unit_num, '(A)') "# 1D Heat Diffusion Solution"
    write(unit_num, '(A,I8)')    "# Step: ", step
    write(unit_num, '(A,F12.6)') "# Time: ", t
    write(unit_num, '(A)') "x,u_numerical,u_analytical,error"
    
    ! Write data (ES23.16 for full double precision)
    do i = 1, nx
      write(unit_num, '(ES23.16,A,ES23.16,A,ES23.16,A,ES23.16)') &
            x(i), ",", u(i), ",", u_exact(i), ",", u(i) - u_exact(i)
    end do
    
    close(unit_num)
  end subroutine write_solution

  subroutine write_history(step, t, l2_error, max_error, total_heat)
    !-------------------------------------------------------------------------
    ! Append to history file (time series of errors)
    !-------------------------------------------------------------------------
    implicit none
    integer, intent(in) :: step
    real(8), intent(in) :: t
    real(8), intent(in) :: l2_error
    real(8), intent(in) :: max_error
    real(8), intent(in) :: total_heat
    
    character(len=256) :: filename
    integer :: unit_num
    logical :: file_exists
    
    unit_num = 21
    write(filename, '(A,A)') trim(output_dir), "history.csv"
    
    inquire(file=trim(filename), exist=file_exists)
    
    if (.not. file_exists) then
      open(unit=unit_num, file=trim(filename), status='new')
      write(unit_num, '(A)') "step,time,l2_error,max_error,total_heat"
    else
      open(unit=unit_num, file=trim(filename), status='old', position='append')
    end if
    
    write(unit_num, '(I8,A,F12.8,A,ES14.6,A,ES14.6,A,F12.8)') &
          step, ",", t, ",", l2_error, ",", max_error, ",", total_heat
    
    close(unit_num)
  end subroutine write_history

  subroutine write_final_report(total_steps, final_time, l2_error, max_error)
    !-------------------------------------------------------------------------
    ! Write final summary report
    !-------------------------------------------------------------------------
    implicit none
    integer, intent(in) :: total_steps
    real(8), intent(in) :: final_time
    real(8), intent(in) :: l2_error
    real(8), intent(in) :: max_error
    
    character(len=256) :: filename
    integer :: unit_num
    
    unit_num = 22
    write(filename, '(A,A)') trim(output_dir), "report.txt"
    
    open(unit=unit_num, file=trim(filename), status='replace')
    
    write(unit_num, '(A)') "========================================"
    write(unit_num, '(A)') "1D Heat Diffusion - Final Report"
    write(unit_num, '(A)') "========================================"
    write(unit_num, '(A)')
    write(unit_num, '(A)') "PARAMETERS:"
    write(unit_num, '(A,I8)')      "  Grid points:    ", nx
    write(unit_num, '(A,F12.8)')   "  Domain length:  ", L
    write(unit_num, '(A,ES14.6)')  "  dx:             ", dx
    write(unit_num, '(A,ES14.6)')  "  dt:             ", dt
    write(unit_num, '(A,ES14.6)')  "  alpha:          ", alpha
    write(unit_num, '(A,F12.8)')   "  CFL number:     ", cfl
    write(unit_num, '(A)')
    write(unit_num, '(A)') "RESULTS:"
    write(unit_num, '(A,I8)')      "  Total steps:    ", total_steps
    write(unit_num, '(A,F12.8)')   "  Final time:     ", final_time
    write(unit_num, '(A,ES14.6)')  "  L2 error:       ", l2_error
    write(unit_num, '(A,ES14.6)')  "  Max error:      ", max_error
    write(unit_num, '(A)')
    write(unit_num, '(A)') "========================================"
    
    close(unit_num)
    
    ! Also print to console
    write(*,'(A)')
    write(*,'(A)') "Final Results:"
    write(*,'(A,I8)')     "  Total steps:  ", total_steps
    write(*,'(A,F12.8)')  "  Final time:   ", final_time
    write(*,'(A,ES14.6)') "  L2 error:     ", l2_error
    write(*,'(A,ES14.6)') "  Max error:    ", max_error
  end subroutine write_final_report

end module output_mod
