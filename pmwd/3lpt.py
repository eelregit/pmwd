import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import ode

class LPT3(object) :
    
    def __init__(self, box_length, num_mesh_1d, Omega_m, Omega_k=0) :
        '''
            Class for computing 3rd order Lagrangian perturbation theory displacement potentials and displacements up to third order.
            
            box_length: length of one side of cube
            num_mesh_1d: number of divisions along one dimension of cube
        '''
        self.box_length = box_length
        self.num_mesh_1d = num_mesh_1d
        self.Omega_m = Omega_m
        self.Omega_k = Omega_k
        self.Omega_de = 1 - self.Omega_m - self.Omega_k
        self.half_num_mesh_1d = np.uint32(np.floor(num_mesh_1d / 2))
        self.num_modes_last_d = self.half_num_mesh_1d + 1
        self.bin_volume = (self.box_length / self.num_mesh_1d) ** 3
        self.fundamental_mode = 2. * np.pi / self.box_length
        self.wave_numbers = np.fft.fftfreq(self.num_mesh_1d, d = 1. / self.fundamental_mode / self.num_mesh_1d)
        self.field_shape = [self.num_mesh_1d, self.num_mesh_1d, self.num_mesh_1d]
        self.modes_shape = [self.num_mesh_1d, self.num_mesh_1d, self.num_modes_last_d]
        self._initializeGrowthTables()
        
        self.H0 = 1. / 2.99792458e3 # in units of h / Mpc

    def _getESquared(self, a) :
        '''Returns Hubble rate squared in units of H_0^2 at scale factor a.'''
        return self.Omega_m * a ** -3 + self.Omega_k * a ** -2 + self.Omega_de
        
    def _getE(self, a) :
        '''Returns Hubble rate squared in units of H_0^2 at scale factor a.'''
        return np.sqrt(self._getESquared(a))
    
    def _getHSquared(self, a) :
        '''Returns Hubble rate squared in units of (h / Mpc) ** 2 at scale factor a.'''
        return self.H0 ** 2 * self._getESquared(a)
        
    def _getH(self, a) :
        '''Returns Hubble rate in units of h / Mpc at scale factor a.'''
        return self.H0 * self._getE(a)
    
    def _getDiffLogH(self, a) :
        '''Returns d log Hubble / d log a.'''
        return -0.5 * (3 * self.Omega_m * a ** -3 + 2 * self.Omega_k * a ** -2) / self._getESquared(a)
    
    def _getBeta(self, a) :
        '''Returns 1.5 * Omega_m(a) at scale factor a.'''
        return 1.5 * self.Omega_m * a ** -3 / self._getESquared(a)
    
    def _fft(self, field) :
        '''Returns normalized forward Fourier transform of field so the modes has units [field] * [volume].'''
        return self.bin_volume * np.fft.rfftn(field)
    
    def _ifft(self, field) :
        '''Returns normalized backward Fourier transform of field (modes) so the position space field has units [field (modes)] / [volume].'''
        return np.fft.irfftn(field) / self.bin_volume
    
    def _getSafeReciprocal3D(self, field) :
        '''Returns 1 / field where field is in Fourier space except for the zero amplitude modes which remain zero.'''
        field[0,0,0] = 1.
        field = 1. / field
        field[0,0,0] = 0.
        return field
    
    def _getWaveVectorNorms(self) :
        '''Returns 3D array of wave vector norms.'''
        return (self.wave_numbers ** 2 + self.wave_numbers[:, None] ** 2 + self.wave_numbers[:, None, None] ** 2)[:, :, :self.num_modes_last_d]
        
    def getGrad3D(self, field, dir) :
        '''Returns gradient of field (modes) along direction dir (0, 1, or 2).'''
        if dir == 0 :
            return (field.T * 1j * self.wave_numbers).T
        elif dir == 1 :
            return (field.T * 1j * self.wave_numbers[:, None]).T
        elif dir == 2 :
            return field * 1j * (self.wave_numbers[:, None, None])[:, :, :self.num_modes_last_d]
        else :
            raise ValueError('3D Index out of bounds %d' % (ind))
        return

    def getInverseLaplacian3D(self, field) :
        '''Returns inverse Laplacian of field (modes).'''
        return -field * self._getSafeReciprocal3D(self._getWaveVectorNorms())

    def getLaplacian3D(self, field) :
        '''Returns Laplacian of field (modes)'''
        return -field * self._getWaveVectorNorms()

    def getHessian3D(self, field, dirs) :
        '''Returns Hessian component of field (modes) along directions dirs = [dir1, dir2].'''
        if np.shape(dirs) != (2,) :
            raise ValueError('Hessian inds must have shape (2,).')
        return self.getGrad3D(self.getGrad3D(field, dirs[0]), dirs[1])

    def getHessian3DIFFT(self, field, dirs) :
        '''Returns backward Fourier transform of Hessian component of field (modes) along directions dirs = [dir1, dir2].'''
        return self._ifft(self.getHessian3D(field, dirs))
    
    def convolveHessian3D(self, fields, dirs) :
        '''
            Returns to convolution of the Hessian components dirs[i] of every field[i] in fields (modes)
            by multiplying their backward Fourier transforms.
        '''
        if len(fields) != len(dirs) :
            raise ValueError('dirs must be a list with the same length as fields.')
        output = np.ones(self.field_shape)
        for i, phi in enumerate(fields) :
            output *= self.getHessian3DIFFT(phi, dirs[i])
        return output
        
    def convolveHessian3DDifference(self, fields, dirs) :
        '''
            Returns to convolution of the Hessian components dirs[i] of every field[i] in fields (modes)
            by multiplying their backward Fourier transforms, expect for the final two fields whose difference
            field[-2] - field[-1] is convolved instead.
        '''
        if len(fields) != len(dirs) :
            raise ValueError('dirs must be a list with the same length as fields')
        output = np.ones(self.field_shape)
        for i, phi in enumerate(fields[:len(fields) - 2]) :
            output *= self.getHessian3DIFFT(phi, dirs[i])
        output *= self._ifft(self.getHessian3D(fields[-2], dirs[-2]) - self.getHessian3D(fields[-1], dirs[-1]))
        return output
    
    def convolveHessian3DSum(self, fields, dirs) :
        '''
            Returns to convolution of the Hessian components dirs[i] of every field[i] in fields (modes)
            by multiplying their backward Fourier transforms, expect for the final two fields whose sum
            field[-2] + field[-1] is convolved instead.
        '''
        if len(fields) != len(dirs) :
            raise ValueError('dirs must be a list with the same length as fields.')
        output = np.ones(self.field_shape)
        for i, phi in enumerate(fields[:len(fields) - 2]) :
            output *= self.getHessian3DIFFT(phi, dirs[i])
        output *= self._ifft(self.getHessian3D(fields[-2], dirs[-2]) + self.getHessian3D(fields[-1], dirs[-1]))
        return output
    
    def getLinearDelta(self, k, p, sphere_mode = True) :
        '''
            Returns a random Gaussian realization of the linear density contrast in Fourier space,
            sampled from the power spectrum interpolation table (k, p).

            k: array of wave numbers for interpolate, in units of box_length ** -1
            p: array of power spectrum values at k, in units of k ** -3
            sphere_mode: if True sets all modes with k > nyquist to zero
        '''
        sigma = np.sqrt(p * self.box_length ** 3)
        k_grid = np.sqrt(self._getWaveVectorNorms())
        noise = np.fft.rfftn(np.random.randn(self.num_mesh_1d, self.num_mesh_1d, self.num_mesh_1d)) / self.num_mesh_1d ** 1.5
        if sphere_mode :
            nyqiust_mode = self.fundamental_mode * self.half_num_mesh_1d
            noise[k_grid >= nyqiust_mode] = 0.
            k_grid[k_grid >= nyqiust_mode] = 0.
        noise[k_grid > 0.] *= np.exp(interp1d(np.log(k), np.log(sigma))(np.log(k_grid[k_grid > 0.])))
        return noise
    
    def _initializeGrowthTables(self, zi = 1.e6, zf = -0.01, num_z = 1024, nstep = 100, rtol = 1.e-3, atol = 1.e-10) :
        '''
            Numerically intergrates the growth equations for LPT growth factors up to 3rd order, initializes the internal interpolation tables
            for all growth factors and growth rates.
        '''
        #
        # Log scale factors for interpolation tables
        #
        self.log_a = np.linspace(-np.log(1. + zi), -np.log(1. + zf), 1024)
        #
        # Initial conditions assuming matter domination (radiation is ignored)
        #
        a = np.exp(self.log_a[0])
        D1 = a - 2 / 77 * (22 * a ** 2 * self.Omega_k + 7 * a ** 4 * self.Omega_de) / self.Omega_m
        dD1 = a - 8 / 77 * (11 * a ** 2 * self.Omega_k + 7 * a ** 4 * self.Omega_de) / self.Omega_m
        D2 = 3 / 7 * a ** 2 - 1 / 3003 * (1430 * a ** 3 * self.Omega_k + 459 * a ** 5 * self.Omega_de) / self.Omega_m
        dD2 = 6 / 7 * a ** 2 - 1 / 3003 * (4290 * a ** 3 * self.Omega_k + 2295 * a ** 5 * self.Omega_de) / self.Omega_m
        D3a = 1 / 14 * a ** 3 - 2 / 525525 * (30875 * a ** 4 * self.Omega_k + 9933 * a ** 6 * self.Omega_de) / self.Omega_m
        dD3a = 3 / 14 * a ** 3 - 4 / 525525 * (61750 * a ** 4 * self.Omega_k + 29799 * a ** 6 * self.Omega_de) / self.Omega_m
        D3b = 1 / 6 * a ** 3 - 1 / 5775 * (1600 * a ** 4 * self.Omega_k + 511 * a ** 6 * self.Omega_de) / self.Omega_m
        dD3b = 1 / 2 * a ** 3 - 1 / 5775 * (6400 * a ** 4 * self.Omega_k + 3066 * a ** 6 * self.Omega_de) / self.Omega_m
        D3c = 1 / 7 * a ** 3 - 1 / 3003 * (715 * a ** 4 * self.Omega_k + 228 * a ** 6 * self.Omega_de) / self.Omega_m
        dD3c = 3 / 7 * a ** 3 - 4 / 3003 * (715 * a ** 4 * self.Omega_k + 342 * a ** 6 * self.Omega_de) / self.Omega_m
        
        initials = [D1, dD2, D2, dD2, D3a, dD3a, D3b, dD3b, D3c, dD3c]
        #
        # ODEs for LPT growth factors up to order 3 assuming flat LCDM
        #
        def _get_deqs(t_log_a, t_state) :
            D1, dD1, D2, dD2, D3a, dD3a, D3b, dD3b = t_state
            t_a = np.exp(t_log_a)
            beta = self._getBeta(t_a)
            alpha = 2. + self._getDiffLogH(t_a)
            deqs = np.zeros(10)
            deqs[0] = dD1
            deqs[1] = alpha * dD1 + beta * D1
            deqs[2] = dD2
            deqs[3] = alpha * dD2 + beta * (D2 + D1 ** 2)
            deqs[4] = dD3a
            deqs[5] = alpha * dD3a + beta * (D3a + D1 ** 3)
            deqs[6] = dD3b
            deqs[7] = alpha * dD3b + beta * (D3b + D1 * D2)
            deqs[8] = dD3c
            deqs[9] = alpha * dD3c + beta * D1 ** 3
            return deqs
        #
        # Initialize integrator
        #
        growth_odes = ode(_get_deqs)
        growth_odes.set_integrator('lsoda', nsteps = nstep, rtol = rtol, atol = atol)
        self.growth_factors = [initials[::2]]
        self.growth_rates = [initials[1::2]]
        growth_odes.set_initial_value(initials, self.log_a[0])
        #
        # Loop through log(a) array to build interpolation tables
        #
        for t_log_a in self.log_a[1:] :
            growth_odes.integrate(t_log_a)
            if not growth_odes.successful() :
                if not self.quiet :
                    warnings.warn("LPT3.initializeGrowthTables failed to integrate at log(a) = %.6e." % (growth_odes.t))
                break
            self.growth_factors.append(growth_odes.y[::2])
            self.growth_rates.append(self._getH(np.exp(t_log_a)) * growth_odes.y[1::2])
        #
        # Define interal interpolators
        #
        self.growth_factors = np.array(self.growth_factors).T
        self.growth_rates = np.array(self.growth_rates).T
        self._getD1 = interp1d(self.log_a, self.growth_factors[0])
        self._getD2 = interp1d(self.log_a, self.growth_factors[1])
        self._getD3a = interp1d(self.log_a, self.growth_factors[2])
        self._getD3b = interp1d(self.log_a, self.growth_factors[3])
        self._getD3c = interp1d(self.log_a, self.growth_factors[4])
        self._getdD1 = interp1d(self.log_a, self.growth_rates[0])
        self._getdD2 = interp1d(self.log_a, self.growth_rates[1])
        self._getdD3a = interp1d(self.log_a, self.growth_rates[2])
        self._getdD3b = interp1d(self.log_a, self.growth_rates[3])
        self._getdD3c = interp1d(self.log_a, self.growth_rates[4])         

        return
    
    def getD1(self, z) :
        '''Returns first order growth factor interpolated at redshift z.'''
        return self._getD1(-np.log(1. + z))

    def getD2(self, z) :
        '''Returns second order growth factor interpolated at redshift z.'''
        return self._getD2(-np.log(1. + z))

    def getD3a(self, z) :
        '''Returns third order growth factor sourced by D1 ** 3 interpolated at redshift z.'''
        return self._getD3a(-np.log(1. + z))

    def getD3b(self, z) :
        '''Returns third order growth factor sourced by D1 * D2 interpolated at redshift z.'''
        return self._getD3b(-np.log(1. + z))

    def getD3c(self, z) :
        '''Returns transverse third order growth factor sourced by D1 ** 3 interpolated at redshift z.'''
        return self._getD3c(-np.log(1. + z))
    
    def getdD1(self, z) :
        '''Returns first order growth rate interpolated at redshift z.'''
        return self._getdD1(-np.log(1. + z))

    def getdD2(self, z) :
        '''Returns second order growth rate interpolated at redshift z.'''
        return self._getdD2(-np.log(1. + z))

    def getdD3a(self, z) :
        '''Returns third order growth rate sourced by D1 ** 3 interpolated at redshift z.'''
        return self._getdD3a(-np.log(1. + z))

    def getdD3b(self, z) :
        '''Returns third order growth rate sourced by D1 * D2 interpolated at redshift z.'''
        return self._getdD3b(-np.log(1. + z)) 

    def getdD3c(self, z) :
        '''Returns transverse third order growth rate sourced by D1 ** 3 interpolated at redshift z.'''
        return self._getdD3c(-np.log(1. + z)) 
    
    def getDisplacements(self, delta, z, t_order = 3, z_delta = None):
        '''
            Computes phi and A, the 3rd order LPT displacement scalar and vector potentials respectively at z.
            
            Returns dis, vel, the t_order displacement and velocit fields.
            
            delta: 3D Fourier modes of the linear density contrast field, if not computed at redshift z then z_delta must be given
            z: reshift of desired LPT displacements, if z_delta is None this must also be the redshift of the linear field delta
            t_order: order of Lagrangian perturbations theory (max is 3)
            z_delta: redshift of delta if different from z
        '''
        #
        # 1LPT
        #
        if z_delta is None :
            phi_1 = self.getInverseLaplacian3D(delta) / self.getD1(z)
        else :
            phi_1 = self.getInverseLaplacian3D(delta) / z_delta
        phi_dis = self.getD1(z) * phi_1
        phi_vel = self.getdD1(z) * phi_1
        if t_order > 1 :
            #
            # 2LPT
            #
            phi_2 = self.convolveHessian3DSum([phi_1, phi_1, phi_1], [[0, 0], [1, 1], [2, 2]])
            phi_2 += self.convolveHessian3D([phi_1, phi_1], [[1, 1], [2, 2]])
            for i in range(3) :
                j = (i + 1) % 3
                phi_2 -= self.convolveHessian3D([phi_1, phi_1], [[i, j], [i, j]])
            phi_2 = self.getInverseLaplacian3D(self._fft(phi_2))
            phi_dis += self.getD2(z) * phi_2
            phi_vel += self.getdD2(z) * phi_2
            if t_order > 2 :
                #
                # 3LPT Scalar
                #
                phi_3 = np.zeros(self.field_shape)
                for i in range(3) :
                    j = (i + 1) % 3
                    k = (i + 2) % 3
                    phi_3 += self.convolveHessian3DSum([phi_1, phi_2, phi_2], [[i, i], [j, j], [k, k]])
                    phi_3 -= 2. * self.convolveHessian3D([phi_1, phi_2], [[i, j], [i, j]])
                phi_dis += self.getD3b(z) * self.getInverseLaplacian3D(self._fft(phi_3))
                phi_vel += self.getdD3b(z) * self.getInverseLaplacian3D(self._fft(phi_3))
                phi_3 += self.convolveHessian3D([phi_1, phi_1, phi_1], [[0, 0], [1, 1], [2, 2]])
                phi_3 += 2. * self.convolveHessian3D([phi_1, phi_1, phi_1], [[0, 1], [1, 2], [2, 0]])
                for i in range(3) :
                    j = (i + 1) % 3
                    k = (i + 2) % 3 
                    phi_3 -= self.convolveHessian3D([phi_1, phi_1, phi_1], [[i, i], [j, k], [j, k]])
                phi_3 = self.getInverseLaplacian3D(self._fft(phi_3))
                phi_dis += self.getD3a(z) * phi_3
                phi_vel += self.getdD3a(z) * phi_3
                del(phi_3)
                #
                #3 LPT Vector
                #
                A = np.zeros([3] + self.field_shape)
                for i in range(3) :
                    j = (i + 1) % 3
                    k = (i + 2) % 3
                    A[i] += self.convolveHessian3D([phi_2, phi_1], [[i, j], [i, k]])
                    A[i] -= self.convolveHessian3D([phi_1, phi_2], [[i, j], [i, k]])
                    A[i] += self.convolveHessian3DDifference([phi_1, phi_2, phi_2], [[j, k], [j, j], [k, k]])
                    A[i] -= self.convolveHessian3DDifference([phi_2, phi_1, phi_1], [[j, k], [j, j], [k, k]])
                del(phi_1)
                del(phi_2)
                A = np.array([self.getInverseLaplacian3D(self._fft(t_A)) for t_A in A])
        #
        # Convert potentials to displacements, Note only the vector potential hasn't been scaled by its growth factor/rate
        #
        dis = []
        vel = []
        for i in range(3) :
            t_dis = -self._ifft(self.getGrad3D(phi_dis, i))
            t_vel = -self._ifft(self.getGrad3D(phi_vel, i))
            if t_order > 2 :
                j = (i + 1) % 3
                k = (i + 2) % 3
                curl = self._ifft(self.getGrad3D(A[k], j) - self.getGrad3D(A[j], k))
                t_dis += self.getD3c(z) * curl
                t_vel += self.getdD3c(z) * curl
            dis.append(t_dis)
            vel.append(t_vel)
        return np.array(dis), np.array(vel)
