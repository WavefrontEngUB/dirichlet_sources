program dirichlet
    use :: m_rw_diff, only: compute_focal_field
    use :: m_npy, only: save_npy
    use, intrinsic :: iso_c_binding, only: c_double, c_double_complex, c_int, c_null_char
    use, intrinsic :: iso_fortran_env, only: stdout=>output_unit, int64, stderr=>error_unit
    
    implicit none

    ! Paràmetres
    real(c_double), parameter :: pi = transfer(int(z'400921FB54442D18', kind=int64), 1.0_c_double)
    real(c_double), parameter :: lambda = 500e-6_c_double   ! mm
    real(c_double), parameter :: f = 5.0_c_double/lambda
    real(c_double), parameter :: n = 1.0_c_double
    real(c_double), parameter :: j1argmax = 1.8411837813406593_c_double ! Posició del primer màxim j1
    complex(c_double_complex), parameter :: I_u = cmplx(0.0_c_double, 1.0_c_double, kind=c_double)
    ! TODO: Calcula les fonts radial i azimutal
    integer :: n_modes, n_lat, n_z, i, j, idx, bess_osc
    real(c_double) :: NA, z, l_focal, length, dz, z_i
    character(255) :: fname, mode

    ! Camps d'entrada i sortida
    complex(c_double_complex), allocatable :: Ein(:, :, :), Eout(:, :, :, :)
    real(c_double), allocatable :: dop(:, :, :, :), irr(:, :, :, :)

    ! Llegim la configuració des dels arguments
    call get_arguments(n_lat, n_modes, n_z, z, l_focal, NA, mode, bess_osc)
    ! Calculem la longitud de la finestra de sampling a la PE del sistema òptic
    length = f*n_lat/4/l_focal
    ! Separació entre plans
    dz = (2*z)/n_z

    ! Reservem espai per als arrays necessaris
    allocate(Ein(n_lat, n_lat, 2), Eout(n_lat, n_lat, 3, 2*n_modes+1), &
             dop(n_lat, n_lat, n_modes+1, n_z+1), irr(n_lat, n_lat, n_modes+1, n_z+1))

    Ein  = 0
    Eout = 0
    dop  = 0
    irr  = 0

    do j = 1, n_z + 1
        z_i = -z+dz*(j-1)
        write(stdout, "(A, G0.3)") "Computing z = ", z_i
        ! Calcula, per a cada z, els modes del camp parcialment coherent
        do i = 0, n_modes
            if (i == 0) then
                call create_pol_field(trim(mode), i, Ein, length/f/NA, bess_osc)
                call compute_focal_field(Ein, Eout(:, :, :, 1), [length, length], f, n, NA, z_i)
                call compute_dop_irr(dop(:, :, 1, j), irr(:, :, 1, j), Eout(:, :, :, 1:1))
            else
                idx = 2*i
                ! N = i
                call create_pol_field(trim(mode), i, Ein, length/f/NA, bess_osc)
                call compute_focal_field(Ein, Eout(:, :, :, idx), [length, length], f, n, NA, z_i)

                ! N = -i
                call create_pol_field(trim(mode), -i, Ein, length/f/NA, bess_osc)
                call compute_focal_field(Ein, Eout(:, :, :, idx+1), [length, length], f, n, NA, z_i)
                ! Els camps s'ordenen segons 
                ! E0    idx = 1
                ! E1    idx = 2
                ! E-1   idx = 3
                ! E2    idx = 4
                ! E-2   idx = 5
                ! ...
                call compute_dop_irr(dop(:, :, i+1, j), irr(:, :, i+1, j), Eout(:, :, :, 1:idx+1))
            end if
        end do
    end do

    ! Desem els resultats...
    write(stdout, "(A)") "Saving results..."
    write(fname, "(A, A)") trim(mode)
    call system("mkdir "//trim(fname))
    write(fname, "(A, A)") trim(mode), "/npys"
    call system("mkdir "//trim(fname))
    do i = 0, n_modes
        write(stdout, "(I0)") i
        write(fname, "(A, A, I0, A, A, A)") trim(mode), "/npys/", i, "_", trim(mode), "-dop.npy"
        call save_npy(fname, dop(:, :, i+1, :))
        write(fname, "(A, A, I0, A, A, A)") trim(mode), "/npys/", i, "_", trim(mode), "-irr.npy"
        call save_npy(fname, irr(:, :, i+1, :))
    end do

    ! Cleanup
    deallocate(Ein, Eout, irr, dop)

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    ! Subrutines utilitzades
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    contains
        subroutine get_arguments(n_lat, n_modes, n_z, z, l_focal, NA, mode, bess_osc)
            integer, intent(out) :: n_lat, n_modes, n_z, bess_osc
            real(c_double), intent(out) :: z, l_focal, NA
            character(*), intent(out) :: mode

            integer :: argc
            character(255) :: argv
            ! Get command line arguments
            argc = command_argument_count()

            if (argc < 6) then
                write(stdout, "(A)") "Usage: dirichlet [N_lateral] [N_modes] [z] [l_focal] [NA]"
                write(stdout, "(A)") "    - N_lateral: lateral number of sampling points"
                write(stdout, "(A)") "    - N_modes: lateral number of sampling points"
                write(stdout, "(A)") "    - n_z: number of z planes to compute"
                write(stdout, "(A)") "    - z: maximum distance from focus"
                write(stdout, "(A)") "    - l_focal: Half size of the focal region window"
                write(stdout, "(A)") "    - NA: Numerical aperture"
                write(stdout, "(A)") "    - mode: Polarization mode, radial or azimuthal (optional, defaults radial)"
                stop
            end if

            ! Llegim arguments un a un
            call get_command_argument(1, argv)
            read(argv, *) n_lat
            call get_command_argument(2, argv)
            read(argv, *) n_modes
            call get_command_argument(3, argv)
            read(argv, *) n_z
            call get_command_argument(4, argv)
            read(argv, *) z
            call get_command_argument(5, argv)
            read(argv, *) l_focal
            call get_command_argument(6, argv)
            read(argv, *) NA
            if (argc >= 7) then
                call get_command_argument(7, argv)
                read(argv, *) mode
            else
                mode = "radial"
            end if
            if (argc == 8) then
                call get_command_argument(8, argv)
                read(argv, *) bess_osc
            else
                bess_osc = 0
            end if

            ! Donem la config per stderr
            write(stderr, "(A)") "Using: "
            write(stderr, "(A, I5)")    "    N_lateral = ", n_lat
            write(stderr, "(A, I5)")    "    N_modes   = ", n_modes
            write(stderr, "(A, I5)")    "    N_z       = ", n_z
            write(stderr, "(A, G10.3)") "    z         = ", z
            write(stderr, "(A, G10.3)") "    l_focal   = ", l_focal
            write(stderr, "(A, G10.3)") "    NA        = ", NA
            write(stderr, "(A, G10.3)") "    mode      = ", mode
            write(stderr, "(A, I5)")    "    bess_osc  = ", bess_osc

        end subroutine get_arguments

        subroutine create_pol_field(polarization, mode, Ein, ctant, bess_osc)
            character(*), intent(in) :: polarization
            integer, intent(in) :: mode, bess_osc
            real(c_double), intent(in) :: ctant
            complex(c_double_complex), intent(out) :: Ein(:, :, :)

            integer :: dims(3), i, j
            real(c_double) :: x, y, phi, r, dx, dy
            dims = shape(Ein)
            
            !$omp parallel do simd private(r, y, x, phi)
            do j = 1, dims(2)
                y = -dims(2)/2+(j-1)
                do i = 1, dims(1)
                    x = -dims(1)/2+(i-1)
                    phi = datan2(y, x)
                    r = dsqrt(x*x+y*y)
                    if (polarization == "radial") then
                        Ein(i, j, 1) = r*dcos(phi)
                        Ein(i, j, 2) = r*dsin(phi)
                    else if (polarization == "azimuthal") then
                        Ein(i, j, 1) = -r*dsin(phi)
                        Ein(i, j, 2) =  r*dcos(phi)
                    end if
                    ! FIXME: Afegit bessel_j1 per desig de la Charo Idus Martius MMXXIII
                    Ein(i, j, 1) = Ein(i, j, 1)*cdexp(I_u*phi*mode)/(dims(1)*dsqrt(2.0_c_double))
                    Ein(i, j, 2) = Ein(i, j, 2)*cdexp(I_u*phi*mode)/(dims(1)*dsqrt(2.0_c_double)) 
                    if (bess_osc > 0) then
                        Ein(i, j, 1) = Ein(i, j, 1)*bessel_j1(r/real(dims(2),kind=c_double)*ctant*2*bess_osc)
                        Ein(i, j, 2) = Ein(i, j, 2)*bessel_j1(r/real(dims(2),kind=c_double)*ctant*2*bess_osc)
                    end if
                end do
            end do
            !$omp end parallel do simd
        end subroutine create_pol_field

        subroutine compute_dop_irr(dop, irr, En)
            !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            ! Càlcul del dop i la irradiància segons el conjunt
            ! de modes del camp En.
            !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            real(c_double), intent(out) :: dop(:, :), irr(:, :)
            complex(c_double_complex), intent(in) :: En(:, :, :, :)

            integer :: dims(4), i, j, k
            complex(c_double_complex), allocatable :: W(:, :, :, :)
            
            dims = shape(En)
            allocate(W(dims(1), dims(2), 3, 3))
            W = 0
            do k = 1, dims(4)   ! Per a cada mode
                !!$omp parallel do simd
                do j = 1, dims(2)   ! Per a cada punt
                    do i = 1, dims(1)
                        ! Per a cada mode, sumem la contribució de la irradiància!
                        W(i, j, :, :) = W(i, j, :, :) + &
                                        matmul(reshape(En(i, j, :, k), [3, 1]), &
                                               reshape(dconjg(En(i, j, :, k)), [1, 3]))

                        !!W(i, j, 1, 1) = W(i, j, 1, 1)+En(i, j, 1, k)*dconjg(En(i, j, 1, k))
                        !!W(i, j, 1, 2) = W(i, j, 1, 2)+dconjg(En(i, j, 1, k))*En(i, j, 2, k)
                        !!W(i, j, 1, 3) = W(i, j, 1, 3)+dconjg(En(i, j, 1, k))*En(i, j, 3, k)

                        !W(i, j, 2, 1) = W(i, j, 2, 1)+dconjg(En(i, j, 2, k))*En(i, j, 1, k)
                        !!W(i, j, 2, 1) = W(i, j, 2, 1)+dconjg(W(i, j, 1, 2))
                        !!W(i, j, 2, 2) = W(i, j, 2, 2)+En(i, j, 2, k)*dconjg(En(i, j, 2, k))
                        !!W(i, j, 2, 3) = W(i, j, 2, 3)+dconjg(En(i, j, 2, k))*En(i, j, 3, k)

                        !W(i, j, 3, 1) = W(i, j, 3, 1)+dconjg(En(i, j, 3, k))*En(i, j, 1, k)
                        !W(i, j, 3, 2) = W(i, j, 3, 2)+dconjg(En(i, j, 3, k))*En(i, j, 2, k)
                        !!W(i, j, 3, 1) = W(i, j, 3, 1)+dconjg(W(i, j, 1, 3))
                        !!W(i, j, 3, 2) = W(i, j, 3, 2)+dconjg(W(i, j, 2, 3))
                        !!W(i, j, 3, 3) = W(i, j, 3, 3)+dconjg(En(i, j, 3, k))*En(i, j, 3, k)

                    end do 
                end do
                !!$omp end parallel do simd
            end do

            ! Calculem la irradiància primer de tot
            call compute_irradiance(irr, W)
            ! Calculem el dop
            call compute_dop(dop, irr, W)
            deallocate(W)
        end subroutine compute_dop_irr

        subroutine compute_irradiance(irr, W)
            real(c_double), intent(out) :: irr(:, :)
            complex(c_double_complex), intent(in) :: W(:, :, :, :)
            
            integer :: dims(4), i, j
            dims = shape(W)

            !$omp parallel do simd
            do j = 1, dims(2)
                do i = 1, dims(1)
                    irr(i, j) = W(i, j, 1, 1)+&
                                W(i, j, 2, 2)+&
                                W(i, j, 3, 3)
                end do
            end do
            !$omp end parallel do simd
        end subroutine compute_irradiance

        pure complex function trace(matrix) 
            complex(c_double_complex), intent(in) :: matrix(3, 3)
            trace = matrix(1, 1)+matrix(2, 2)+matrix(3,3)
        end function trace

        subroutine compute_dop(dop, irr, W)
            complex(c_double_complex), intent(in) :: W(:, :, :, :) 
            real(c_double), intent(in)  :: irr(:, :)
            real(c_double), intent(out) :: dop(:, :)

            real(c_double), allocatable :: trw2(:, :)
            integer :: dims(2), i, j

            dims = shape(irr)
            allocate(trw2(dims(1), dims(2)))

            ! Expandeixo la suma...
            ! Canvi 21/09/2022
            trw2 = sum(sum(W*dconjg(W), dim=4), dim=3)

            dop = 1.5_c_double*trw2/(irr*irr) - 0.5_c_double
            deallocate(trw2)
        end subroutine compute_dop

end program dirichlet
