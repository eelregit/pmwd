import numpy as np

class LPT3(object) :
    
    def __init__(self, box_length, num_mesh_1d) :
        '''
            Class for computing 3rd order Lagrangian perturbation theory displacement potentials and displacements
            
            box_length: length of one side of cube
            num_mesh_1d: number of divisions along one dimension of cube
        '''
        self.box_length = box_length
        self.num_mesh_1d = num_mesh_1d
        self.half_num_mesh_1d = np.uint32(np.floor(num_mesh_1d / 2))
        self.num_modes_last_d = self.half_num_mesh_1d + 1
        self.bin_volume = (self.box_length / self.num_mesh_1d) ** 3
        self.fundamental_mode = 2. * np.pi / self.box_length
        self.wave_numbers = np.fft.fftfreq(self.num_mesh_1d, d = 1. / self.fundamental_mode / self.num_mesh_1d)
        self.field_shape = [self.num_mesh_1d, self.num_mesh_1d, self.num_mesh_1d]
        self.modes_shape = [self.num_mesh_1d, self.num_mesh_1d, self.num_modes_last_d]

    def _fft(self, field) :
        return self.bin_volume * np.fft.rfftn(field)
    
    def _ifft(self, field) :
        return np.fft.irfftn(field) / self.bin_volume
    
    def _getSafeReciprocal3D(self, field) :
        field[0,0,0] = 1.
        field = 1. / field
        field[0,0,0] = 0.
        return field
    
    def _getWaveVectorNorms(self) :
        return (self.wave_numbers ** 2 + self.wave_numbers[:, None] ** 2 + self.wave_numbers[:, None, None] ** 2)[:, :, :self.num_modes_last_d]
        
    def getGrad3D(self, field, ind) :
        if ind == 0 :
            return (field.T * 1j * self.wave_numbers).T
        elif ind == 1 :
            return (field.T * 1j * self.wave_numbers[:, None]).T
        elif ind == 2 :
            return field * 1j * (self.wave_numbers[:, None, None])[:, :, :self.num_modes_last_d]
        else :
            raise ValueError('3D Index out of bounds %d' % (ind))
        return

    def getInverseLaplacian3D(self, field) :
        return -field * self._getSafeReciprocal3D(self._getWaveVectorNorms())

    def getLaplacian3D(self, field) :
        return -field * self._getWaveVectorNorms()

    def getHessian3D(self, field, inds) :
        if np.shape(inds) != (2,) :
            raise ValueError('Hessian inds must have shape (2,)')
        return self.getGrad3D(self.getGrad3D(field, inds[0]), inds[1])

    def getHessian3DIFFT(self, field, inds) :
        return self._ifft(self.getHessian3D(field, inds))
    
    def convolveHessian3D(self, fields, inds) :
        if len(fields) != len(inds) :
            raise ValueError('inds must be a list with the same length as fields')
        output = np.ones(self.field_shape)
        for i, phi in enumerate(fields) :
            output *= self.getHessian3DIFFT(phi, inds[i])
        return output
        
    def convolveHessian3DDifference(self, fields, inds) :
        if len(fields) != len(inds) :
            raise ValueError('inds must be a list with the same length as fields')
        output = np.ones(self.field_shape)
        for i, phi in enumerate(fields[:len(fields) - 2]) :
            output *= self.getHessian3DIFFT(phi, inds[i])
        output *= self._ifft(self.getHessian3D(fields[-2], inds[-2]) - self.getHessian3D(fields[-1], inds[-1]))
        return output
    
    def getLinearDelta(self, k, p, sphere_mode = True) :
        '''
            Returns a random Gaussian realization of the linear density contrast in Fourier space,
            sampled from the power spectrum interpolation table (k, p)

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

    def get3LPTPotentials(self, delta, z):
        '''
            Returns phi and A, the 3rd order LPT displacement scalar and vector potentials respectively at z 
            
            delta: 3D Fourier modes of the 1LPT density contrast field at redshit z
            z: reshift of delta and of the output potentials
        '''
        # LPT growth factors
        D1 = 1. / z + 1
        D2 = 3. / 7. * D1 ** 2
        D3_L = 1. / 3. * D1 ** 3
        D3_T = 1. / 7. * D1 ** 3
        # 1LPT
        phi1 = delta / D1
        phi1 = self.getInverseLaplacian3D(delta)
        phi = D1 * phi1
        # 2LPT
        phi2 = np.zeros(self.field_shape)
        for i in range(3) :
            j = (i + 1) % 3
            phi2 += self.convolveHessian3DDifference([phi1, phi1, phi1], [[i, i], [j, j], [i, j]])
        phi2 = self.getInverseLaplacian3D(self._fft(phi2))
        phi += D2 * phi2
        # 3LPT Scalar
        phi3 = self.convolveHessian3D([phi1, phi1, phi1], [[0, 0], [1, 1], [2, 2]])
        phi3 += 2. * self.convolveHessian3D([phi1, phi1, phi1], [[0, 1], [1, 2], [2, 0]])
        for i in range(3) :
            j = (i + 1) % 3
            k = (i + 2) % 3
            phi3 -= self.getHessian3DIFFT(phi1, [i, i]) * (self.getHessian3DIFFT(phi1, [j, k]) ** 2 
                                                  +  5. / 7. * self._ifft(self.getHessian3D(phi2, [j, j]) + self.getHessian3D(phi2, [k, k])))
            phi3 += 10. / 7. * self.convolveHessian3D([phi1, phi2], [[i, j], [i, j]])
        phi3 = self.getInverseLaplacian3D(self._fft(phi3))
        phi += D3_L * phi3
        del(phi3)
        #3 LPT Vector
        A = np.zeros([3] + self.field_shape)
        for i in range(3) :
            j = (i + 1) % 3
            k = (i + 2) % 3
            A[i] += self.convolveHessian3D([phi2, phi1], [[i, j], [i, k]])
            A[i] -= self.convolveHessian3D([phi1, phi2], [[i, j], [i, k]])
            A[i] += self.convolveHessian3DDifference([phi1, phi2, phi2], [[j, k], [j, j], [k, k]])
            A[i] -= self.convolveHessian3DDifference([phi2, phi1, phi1], [[j, k], [j, j], [k, k]])
        A = np.array([self.getInverseLaplacian3D(self._fft(t_A)) for t_A in A])
        A *= D3_T
        return phi, A

    def getDisplacement(self, phi, A = None) :
        '''
            Returns the displacement field in real space

            phi: scalar LPT displacemental potential modes
            A:   vector LPT displacement potential modes
        '''
        psi = []
        for i in range(3) :
            t_psi = self._ifft(self.getGrad3D(phi, i))
            if A is not None :
                j = (i + 1) % 3
                k = (i + 2) % 3
                t_psi += self._ifft(self.getGrad3D(A[k], j) - self.getGrad3D(A[j], k))
            psi.append(t_psi)
        return np.array(psi)
