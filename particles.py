import torch
from torch import cos, sin, abs


class Particles:
    """ 
    Класс частиц

    [enegries] = eV

    [direction] = cm

    [coordinates] = cm
    """

    processes = []

    def __init__(self, energy, direction, coordinates):
        self._energy = torch.as_tensor(energy)
        self._direction = torch.as_tensor(direction)
        self._coordinates = torch.as_tensor(coordinates)
        self._distance_traveled = torch.zeros_like(self._energy)

    def move(self, distance):
        """ Переместить частицы """
        self._distance_traveled += distance
        self._coordinates += self._direction*torch.column_stack([distance]*3)

    def rotate(self, theta, phi, mask):
        """
        Изменить направления частиц

        [theta] = radian
        [phi] = radian
        """
        cos_theta = cos(theta)
        sin_theta = sin(theta)
        delta1 = sin_theta*cos(phi)
        delta2 = sin_theta*sin(phi)
        delta = torch.ones_like(cos_theta) - 2*(self._direction[mask, 2] < 0)
        b = self._direction[mask, 0]*delta1 + self._direction[mask, 1]*delta2
        tmp = cos_theta - b/(1 + abs(self._direction[mask, 2]))
        cos_alpha = self._direction[mask, 0]*tmp + delta1
        cos_beta = self._direction[mask, 1]*tmp + delta2
        cos_gamma = self._direction[mask, 2]*cos_theta - delta*b
        self._direction[mask] = torch.column_stack((cos_alpha, cos_beta, cos_gamma))

    def change_energy(self, energy_change, mask):
        """ Измененить энергии частиц """
        self._energy[mask] -= energy_change

    def add(self, particles):
        """ Добавить частицы """
        self._energy= torch.cat([self._energy, particles._energy])
        self._direction = torch.cat([self._direction, particles.direction])
        self._coordinates = torch.cat([self._coordinates, particles.coordinates])
        self._distance_traveled = torch.cat([self._distance_traveled, particles._distance_traveled])

    def delete(self, mask):
        """ Удалить частицы из рассмотрения """
        self._energy = self._energy[~mask]
        self._direction = self._direction[~mask]
        self._coordinates = self._coordinates[~mask]
        self._distance_traveled = self._distance_traveled[~mask]

    def replace(self, particles, mask):
        """ Заменить частицы """
        self._energy[mask] = particles._energy
        self._direction[mask] = particles._direction
        self._coordinates[mask] = particles.coordinates
        self._distance_traveled[mask] = particles._distance_traveled

    @property
    def energy(self):
        """ Энергии частиц """
        return self._energy

    @property
    def direction(self):
        """ Направления частиц """
        return self._direction

    @property
    def coordinates(self):
        """ Координаты частиц """
        return self._coordinates

    @property
    def distance_traveled(self):
        """ Расстояние пройденное частицами """
        return self._distance_traveled

    @property
    def count(self):
        """ Число частиц """
        return self._energy.size()


class Photons(Particles):
    """ 
    Класс фотонов

    [enegries] = eV

    [direction] = cm
    
    [coordinates] = cm

    [emission_time] = sec
    """

    processes = ['PhotoelectricEffect', 'ComptonScattering']

    def __init__(self, energy, direction, coordinates, emission_time):
        super().__init__(energy, direction, coordinates)
        self._emission_time = torch.as_tensor(emission_time)
        self._emission_coordinates = self.coordinates.clone()

    def add(self, particles):
        """ Добавить частицы """
        super().add(particles)
        self._emission_time = torch.cat([self._emission_time, particles._emission_time])
        self._emission_coordinates = torch.cat([self._emission_coordinates,  particles._emission_coordinates])

    def delete(self, mask):
        """ Удалить частицы из рассмотрения """
        super().delete(mask)
        self._emission_time = self._emission_time[~mask]
        self._emission_coordinates = self._emission_coordinates[~mask]

    def replace(self, particles, mask):
        """ Заменить частицы """
        super().replace(particles, mask)
        self._emission_time[mask] = particles._emission_time
        self._emission_coordinates[mask] = particles._emission_coordinates

    @property
    def emission_time(self):
        """ Времена эмиссии частиц """
        return self._emission_time

    @property
    def emission_coordinates(self):
        """ Координаты эмисии """
        return self._emission_coordinates

