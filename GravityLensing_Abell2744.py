from astropy.cosmology import FlatLambdaCDM
import numpy as np
from astropy.constants import G, c

class nfwcirc:
    def __init__(self, co, zl=0.308, zs=2.0, c200=4.0, m200=1e15):
        self.co = co  # Kozmolojik model
        self.zl = zl  # Lens kırmızıya kayması
        self.zs = zs  # Kaynak kırmızıya kayması
        self.m200 = m200  # Lensin toplam kütlesi (M200) [Msun]
        self.c200 = c200  # Konsantrasyon parametresi

        # Kritik yoğunluğu hesapla
        self.rhos = (
            200.0 / 3.0 * (self.co.critical_density(self.zl).to("Msun/Mpc3"))
            * self.c200**3
            / (np.log(1.0 + self.c200) - self.c200 / (1.0 + self.c200))
        )

        # r200 ve rs hesapla
        f200 = 4.0 / 3.0 * np.pi * 200.0 * self.co.critical_density(self.zl).to("Msun/Mpc3").value
        self.r200 = (self.m200 / f200) ** (1.0 / 3.0)  # Mpc cinsinden
        self.rs = self.r200 / self.c200

        # Açısal çap uzaklıklarını hesapla
        self.dl = self.co.angular_diameter_distance(self.zl).to("Mpc")
        self.ds = self.co.angular_diameter_distance(self.zs).to("Mpc")
        self.dls = self.co.angular_diameter_distance_z1z2(self.zl, self.zs).to("Mpc")

        # Kritik yüzey yoğunluğu
        self.sc = self.sigma_crit()

        # Yoğunluk ölçeği
        self.ks = self.rhos.value * self.rs / self.sc

    def sigma_crit(self):
        c2G = (c**2 / G).to("Msun/Mpc")
        factor = c2G / (4 * np.pi)
        return (factor * (self.ds / (self.dl * self.dls))).value

    def kappap(self, r):
        x = r / self.rs
        fx = np.piecewise(
            x,
            [x > 1.0, x < 1.0, x == 1.0],
            [
                lambda x: (1 - (2.0 / np.sqrt(x * x - 1.0) * np.arctan(np.sqrt((x - 1.0) / (x + 1.0))))) / (x**2 - 1),
                lambda x: (1 - (2.0 / np.sqrt(1.0 - x * x) * np.arctanh(np.sqrt((1.0 - x) / (1.0 + x))))) / (x**2 - 1),
                0.0,
            ],
        )
        kappa = 2.0 * self.ks * fx
        return kappa

    def massp(self, r):
        x = r / self.rs
        fx = np.piecewise(
            x,
            [x > 1.0, x < 1.0, x == 1.0],
            [
                lambda x: (2.0 / np.sqrt(x * x - 1.0) * np.arctan(np.sqrt((x - 1.0) / (x + 1.0)))),
                lambda x: (2.0 / np.sqrt(1.0 - x * x) * np.arctanh(np.sqrt((1.0 - x) / (1.0 + x)))),
                0.0,
            ],
        )
        massp = 4.0 * self.ks * (np.log(x / 2.0) + fx)
        return massp

# Kozmolojik model
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

# Abell 2744 parametreleri
z_lens = 0.308
z_source = 2.0
c200 = 4.0
m200 = 2e15  # Güneş kütlesi biriminde (Msun)

# NFW modeli
abell_2744_lens = nfwcirc(cosmo, zl=z_lens, zs=z_source, c200=c200, m200=m200)

# Bir yarıçap seçin (örneğin, r200)
r = abell_2744_lens.r200  # Yarıçap (Mpc)

# Kütleyi hesapla
dimensionless_mass = abell_2744_lens.massp(r)
total_mass = dimensionless_mass * m200

print(f"Abell 2744 için toplam kütle (r = r200): {total_mass:.3e} Msun")