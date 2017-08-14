import yt.utilities.fortran_utils as fpu
import yt
import yt.units as U
import numpy as np


def load_brick(filename, bbox=None, ds=None):
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

    am_unit = (1, 'Msun*Mpc*km/s')
    unit_dict = {
        'particle_identifier': (1, '1'), 'particle_mass': (1e11, 'Msun'),
        'particle_position_x': (1, 'Mpc'), 'particle_position_y': (1, 'Mpc'),
        'particle_position_z': (1, 'Mpc'), 'particle_velocity_x': (1, 'km/s'),
        'particle_velocity_y': (1, 'km/s'), 'particle_velocity_z': (1, 'km/s'),
        'subhalo_level': (1, '1'), 'subhalo_hosthalo': (1, '1'),
        'subhalo_host': (1, '1'),
        'subhalo_number': (1, '1'), 'subhalo_next': (1, '1'),
        'particle_angular_momentum_x': am_unit,
        'particle_angular_momentum_y': am_unit,
        'particle_angular_momentum_z': am_unit,
        'virial_radius': (1, 'Mpc'), 'virial_mass': (1e11, 'Msun'),
        'virial_temp': (1, 'K'),
        'virial_vel': (1, 'km/s'),
        'particle_spin': (1, '1'), 'particle_radius': (1, 'Mpc'),
        'particle_axis_a': (1, 'Mpc'), 'particle_axis_b': (1, 'Mpc'),
        'particle_axis_c': (1, 'Mpc')}

    data = {}
    for key in unit_dict:
        intensity = unit_dict[key][0]
        unit = unit_dict[key][1]
        arr = np.array([halos[i][key] for i in halos]) * intensity
        data[key] = (arr, unit)

    n_ref = 1

    ppx, ppy, ppz = [data['particle_position_%s' % d][0]
                     for d in ('x', 'y', 'z')]

    left = np.array([min(ppx), min(ppy), min(ppz)])
    right = np.array([max(ppx), max(ppy), max(ppz)])

    data['particle_position_x'] = (ppx - left[0]), 'Mpc'
    data['particle_position_y'] = (ppy - left[1]), 'Mpc'
    data['particle_position_z'] = (ppz - left[2]), 'Mpc'

    right -= left
    left -= left

    bbox = np.array([left, right]).T

    ds = yt.load_particles(data, length_unit=U.Mpc, mass_unit=1e11*U.Msun,
                           bbox=bbox, n_ref=n_ref)

    @yt.particle_filter('halos', requires=["particle_mass"],
                        filtered_type='all')
    def is_halo(pfilter, data):
        return data[(pfilter.filtered_type, "particle_mass")] > 0

    ds.add_particle_filter('halos')
    return ds


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
        (('particle_angular_momentum_x',
          'particle_angular_momentum_y',
          'particle_angular_momentum_z'), 3, 'f'),
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
