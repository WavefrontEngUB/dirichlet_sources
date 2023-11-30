module m_rw_diff
    use, intrinsic :: iso_fortran_env, only: int32, real64, int64, error_unit
    use, intrinsic :: iso_c_binding
    implicit none
    include "fftw3.f03"

    complex(c_double_complex), parameter :: I_u = (0.0_c_double, 1.0_c_double)
    real(real64), parameter :: pi = transfer(int(z'400921fb54442d18', kind=int64), 1.0_real64)

    private

    public :: compute_focal_field

    contains
        subroutine compute_focal_field(Ein, Eout, length, f, n, NA, z)
            !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            ! Computa el camp focal. Totes les distàncies han d'estar en
            ! termes de múltiples de longitud d'ona
            !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            complex(c_double_complex), intent(in)  :: Ein(:, :, :)
            complex(c_double_complex), intent(out) :: Eout(:, :, :)
            real(real64), intent(in) :: length(2)
            real(c_double), intent(in) :: f, n, NA, z

            integer(int32) :: dims(3), i, j, num_threads
            real(real64) :: length_f(2)
            real(real64), allocatable :: x(:), y(:)
            real(real64) :: dx, dy, sinthmax, d2, sinth2, x2, y2, r2, costh, sinth, sqcos
            real(real64) :: sinphi, cosphi, phi
            type(c_ptr) :: gsr_ptr, out_ptr, plan
            complex(c_double_complex), pointer :: Egsr(:, :, :), Efoc(:, :, :)
            complex(c_double_complex) :: H
            character(255) :: envar

            dims = shape(Ein)
            dims(3) = 3
            ! Calcula el camp focal, escalat segons lambda

            ! Populem les cordenades
            allocate(x(dims(1)), y(dims(2)))
            dx = 2*length(1)/dims(1)
            dy = 2*length(2)/dims(2)
            !$omp parallel do simd
            do i = 1, dims(1)
                x(i) = -length(1)+dx*(i-1)
            end do
            !$omp end parallel do simd
            !$omp parallel do simd
            do i = 1, dims(2)
                y(i) = -length(2)+dy*(i-1)
            end do
            !$omp end parallel do simd

            ! Preparem el camp a GSR juntament amb el pla de la transformada!
            ! Abans de res, iniciem l'entorn de multiprocessing si s'escau
            i = fftw_init_threads()
            if (i == 0) then
                write(error_unit, "(A)") "FFTW3 Error: cannot create threads"
            else
                ! Primerament recollim info sobre el nombre de fils que permet openmp crear 
                ! i planifiquem la transformada en base a ells.
                call get_environment_variable("OMP_NUM_THREADS", envar, status=j) 
                if (j == 0) then
                    read(envar, *) num_threads
                    call fftw_plan_with_nthreads(num_threads)
                    !write(error_unit, "(A,I3,A)") "Using", num_threads, " threads"
                else
                    write(error_unit, "(A, I2)") "Working single threaded, error: ", j
                end if
            end if
            ! Creem els arrays contigus en memòria
            gsr_ptr = fftw_alloc_complex(int(dims(1)*dims(2)*dims(3), kind=c_size_t))
            out_ptr = fftw_alloc_complex(int(dims(1)*dims(2)*dims(3), kind=c_size_t))
            call c_f_pointer(gsr_ptr, Egsr, dims)
            call c_f_pointer(out_ptr, Efoc, dims)

            ! Preparem les transformades. 
            plan = fftw_plan_many_dft(2_c_int, &
                                      [dims(2), dims(1)], &
                                      3_c_int, &            !howmany
                                      Egsr, &               !in
                                      [dims(2), dims(1)], &          !inembed
                                      1_c_int, &            !istride
                                      dims(1)*dims(2), &    !idist
                                      Efoc, &               !out
                                      [dims(2), dims(1)], &          !onembed
                                      1_c_int, &            !ostride
                                      dims(1)*dims(2), &    !odist
                                      fftw_forward, &
                                      fftw_estimate)
            if (.not. c_associated(plan)) then
                write(error_unit, "(A)") "FFTW3 Error: could not create plan!"
                return
            end if

            ! TODO: Compute GSR 
            sinthmax = NA/n
            !$omp parallel do simd private(y2, phi, sinphi, cosphi, x2, r2, sinth2, costh, sqcos, sinth)
            do j = 1, dims(2)
                    y2 = y(j)**2
                do i = 1, dims(1)
                    phi = datan2(y(j), x(i))
                    sinphi = dsin(phi)
                    cosphi = dcos(phi)
                    x2 = x(i)**2
                    ! Calculem les funcions trigonomètriques sobre la GRS
                    r2 = x2+y2
                    sinth2 = r2/f/f
                    if (sinth2 >= sinthmax*sinthmax) then
                        Egsr(i, j, :) = 0.0_c_double
                        cycle
                    end if
                    sinth = dsqrt(sinth2)
                    costh = dsqrt(1.0_real64-sinth2)
                    sqcos = dsqrt(costh)

                    ! Funció de transferència, desplaçament en z
                    H = cdexp(2.0_c_double*I_u*pi*z*costh)
                    !Egsr(i, j, 1) =  Ein(i, j, 1)/sqcos*(cosphi*cosphi*costh+sinphi*sinphi)&
                    !                -Ein(i, j, 2)/sqcos*(1.0_c_double-costh)*sinphi*cosphi
                    Egsr(i, j, 1) = Ein(i, j, 1)*(sinphi*sinphi+cosphi*cosphi*costh)+&
                                    Ein(i, j, 2)*sinphi*cosphi*(costh-1.0_c_double)
                    Egsr(i, j, 1) = Egsr(i, j, 1)*H/sqcos

                    !Egsr(i, j, 2) =  Ein(i, j, 2)/sqcos*(cosphi*cosphi+sinphi*sinphi*costh)&
                    !                -Ein(i, j, 1)/sqcos*(1.0_c_double-costh)*sinphi*cosphi
                    Egsr(i, j, 2) = Ein(i, j, 2)*(cosphi*cosphi+sinphi*sinphi*costh)+&
                                    Ein(i, j, 1)*sinphi*cosphi*(costh-1.0_c_double)
                    Egsr(i, j, 2) = Egsr(i, j, 2)*H/sqcos

                    Egsr(i, j, 3) = -(Ein(i, j, 1)*cosphi+Ein(i, j, 2)*sinphi)*sinth
                    Egsr(i, j, 3) = Egsr(i, j, 3)*H/sqcos
                end do
            end do
            !$omp end parallel do simd

            ! FIXME: SEGFAULT si OPENMP
            call fftw_execute_dft(plan, Egsr, Efoc)
            Eout = Efoc
            
            call fftw_destroy_plan(plan)

            ! Cleanup
            call fftw_free(gsr_ptr)
            call fftw_free(out_ptr)
            deallocate(x, y)
        end subroutine compute_focal_field
end module m_rw_diff
