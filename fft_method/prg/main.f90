program main
    use, intrinsic :: iso_fortran_env, only: int32, real64, stdout=>output_unit, int64
    use, intrinsic :: iso_c_binding, only: c_dc=>c_double_complex, c_double
    use :: m_rw_diff, only: compute_focal_field
    use :: m_npy, only: save_npy

    implicit none

    complex(c_dc), allocatable :: Ein(:, :, :)
    complex(c_dc), allocatable :: Eout(:, :, :)

    integer :: nx, ny, i, j
    real(c_double) :: NA, r
    real(c_double) :: length(2)
    real(c_double), parameter :: n = 1.0_c_double
    real(c_double), parameter :: f_length(2) = [8.0_c_double, 8.0_c_double]
    real(c_double), parameter :: f = 5.0_c_double
    real(c_double), parameter :: pi = transfer(int(z'400921fb54442d18', kind=int64), f)
    complex(c_dc), parameter :: I_u = cmplx(0._c_dc, 1._c_dc, kind=c_dc)

    ! Recollim les dades d'entrada...
    call get_dims(nx, ny, NA, r)
    write(stdout, "(A,2I5, E0.4, E0.4)") "Using:", nx, ny, NA, r

    ! Creem el camp d'entrada
    !slice => Ein(:, :, 0)
    length(1) = nx*f/4.0_c_double/f_length(1)
    length(2) = ny*f/4.0_c_double/f_length(2)
    allocate(Ein(nx, ny, 2), Eout(nx, ny, 3))
    !call circle(Ein(:, :, 1), r)
    Ein(:, :, 1) = 1
    Ein(:, :, 2) = cmplx(0.0_c_dc, 0.0_c_dc, kind=c_dc)
    !Ein(:, :, 2) = I_u*Ein(:, :, 1)
    !call save_npy("Ein.npy", Ein)

    ! Calcul focal
    call compute_focal_field(Ein, Eout, length, f, n, NA, 0.0_c_double)

    call save_npy("Efoc.npy", Eout)


    contains
        subroutine get_dims(nx, ny, NA, r)
            integer, intent(out) :: nx, ny
            real(c_double), intent(out) :: NA, r

            integer :: argc
            character(32) :: argv

            argc = command_argument_count()
            if (argc /= 4) then
                write(stdout, "(A)") "Usage: main nx ny NA r"
                stop
            end if

            call get_command_argument(1, argv)
            read(argv, *) nx
            call get_command_argument(2, argv)
            read(argv, *) ny
            call get_command_argument(3, argv)
            read(argv, *) NA
            call get_command_argument(4, argv)
            read(argv, *) r
        end subroutine get_dims

        subroutine circle(array, radius)
            complex(c_dc), intent(out) :: array(:, :)
            real(c_double), intent(in) :: radius

            real(c_double) :: r2max, x2, y2, r2
            integer :: i, j
            integer :: dims(2)
            dims = shape(array)
            r2max = radius*radius*minval(dims)**2
            
            !$omp parallel do simd private(x2, y2, r2)
            do j = 1, nx
                y2 = (j-ny/2)**2
                do i = 1, ny
                    x2 = (i-nx/2)**2
                    r2 = x2+y2
                    if (r2 > r2max) cycle
                    array(i, j) = 1_c_dc
                end do
            end do
            !$omp end parallel do simd
        end subroutine circle
end program main
