import yt.utilities.fortran_utils as fpu
import yt
import yt.units as U
import numpy as np


def load_brick(filename, bbox=None):
    """Load a brick file as outputed by HaloFinder.

    You can pass a bbox (in Mpc) to force the box size."""
    with open(filename, "rb") as f:
        # Load headers
        hvals = {}
        attrs = (('nbodies', 1, 'i'),
                 ('particle_mass', 1, 'f'),
                 ('aexp', 1, 'f'),
                 ('omega_t', 1, 'f'),
                 ('age_univ', 1, 'f'),
                 (('nhalos', 'nsubhalos'), 2, 'i'))

        hvals.update(fpu.read_attrs(f, attrs))

        # Load halo data
        halos = {}
        for _ in range(hvals['nhalos']):
            tmp = _read_halo(f)

            halo_id = tmp.get('particle_identifier')
            halos[halo_id] = tmp

        for _ in range(hvals['nsubhalos']):
            tmp = _read_halo(f)

            halo_id = tmp.get('particle_identifier')
            halos[halo_id] = tmp

    # Now converts everything into yt-aware quantities
    def g(key):
        return [halos[i][key] for i in halos]

    data = {}
    for key in ['particle_identifier', 'particle_mass',
                'particle_position_x', 'particle_position_y',
                'particle_position_z', 'particle_velocity_x',
                'particle_velocity_y', 'particle_velocity_z',
                'subhalo_level', 'subhalo_hosthalo', 'subhalo_host',
                'subhalo_number', 'subhalo_next', 'L_x', 'L_y',
                'L_z', 'virial_radius', 'virial_mass', 'virial_temp',
                'virial_vel', 'particle_spin', 'particle_radius',
                'particle_axis_a', 'particle_axis_b',
                'particle_axis_c']:
        data[key] = [halos[i][key] for i in halos]

    n_ref = 1
    ppx, ppy, ppz = [data['particle_position_%s' % d] for d in ('x', 'y', 'z')]

    if bbox is None:
        bbox = np.array([[min(ppx), max(ppx)],
                         [min(ppy), max(ppy)],
                         [min(ppz), max(ppz)]])
    return yt.load_particles(data, length_unit=U.Mpc, mass_unit=1e11*U.Msun,
                             bbox=bbox, n_ref=n_ref)


def _read_halo(f):
    hattrs = {}
    hattrs['particle_number'] = fpu.read_vector(f, 'i')[0]
    hattrs['iparts'] = fpu.read_vector(f, 'i')

    halos_attrs = (
        ('particle_identifier', 1, 'i'),
        ('timestep', 1, 'f'),
        (('subhalo_level', 'subhalo_hosthalo', 'subhalo_host',
          'subhalo_number', 'subhalo_next'), 5, 'i'),
        ('particle_mass', 1, 'f'),
        (('particle_position_x',
          'particle_position_y',
          'particle_position_z'), 3, 'f'),
        (('particle_velocity_x',
          'particle_velocity_y',
          'particle_velocity_z'), 3, 'f'),
        (('L_x', 'L_y', 'L_z'), 3, 'f'),
        (('particle_radius', 'particle_axis_a',
          'particle_axis_b', 'particle_axis_c'), 4, 'f'),
        (('particle_Ek', 'particle_Ep', 'particle_Et'), 3, 'f'),
        ('particle_spin', 1, 'f'),
        (('virial_radius',
          'virial_mass',
          'virial_temp',
          'virial_vel'), 4, 'f'),
        (('rho_0', 'r_c'), 2, 'f'))

    hattrs.update(fpu.read_attrs(f, halos_attrs))

    return hattrs
