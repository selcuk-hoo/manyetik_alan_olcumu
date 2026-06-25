# REFERANS: Systematic errors related to quadrupole misplacement in an all-electric storage ring for proton EDM (arXiv:1709.01208)

> pdftotext ile çıkarıldı (özgünlük karşılaştırması için).

---

                                                        Systematic errors related to quadrupole
                                                     misplacement in an all-electric storage ring for
arXiv:1709.01208v1 [physics.acc-ph] 5 Sep 2017




                                                               proton EDM experiment
                                                                Selçuk Hacıömeroğlu1 and Yannis K. Semertzidis1,2

                                                          1
                                                              IBS, Center for Axion and Precision Physics, Daejeon, 34051, South Korea
                                                                     2 KAIST, Physics Department, Daejeon, 34141, South Korea



                                                                                        August 4, 2021

                                                                                             Abstract
                                                           Misplacement of electrostatic elements can pose false EDM signal in a stor-
                                                       age ring EDM experiment because of coupling between vertical electric field and
                                                       magnetic dipole moment. A vertically misplaced quadrupole introduces electric
                                                       field proportional to its misplacement, changing periodically in the particle’s rest
                                                       frame during the storage. This leads to accumulation of vertical spin component
                                                       at every revolution. We investigated this effect by simulating a proton in an all-
                                                       electric ring with several quadrupole scenarios. It turns out that the misplacement
                                                       of quadrupoles is a critical item to keep under control, for which we propose sev-
                                                       eral methods. These include tuning the frequency of the RF cavity, making use
                                                       of additional correction quadrupoles and using quadrupoles with weaker focusing
                                                       strength.


                                                 1    Introduction
                                                 A storage ring with all-electric elements can be used to probe the electric dipole
                                                 moment (EDM) of protons [1, 9]. The electrostatic deflectors in such a ring both store
                                                 the beams and couple with the EDM to induce a spin precession. The spin precession
                                                 under electric and magnetic fields is governed by the T-BMT equation [2]. With the
                                                 additional EDM terms, it is given as.
                                                                     "                                                      !
                                                         d~s    e
                                                                                  
                                                                                1 ~       γG ~ ~ ~               1 β~ ×E ~
                                                             = ~s ×        G+       B−        β(β · B) − G +
                                                          dt    m               γ        γ+1                   γ+1       c
                                                                                                      !#                             (1)
                                                                     η ~        γ ~ ~ ~          ~ ×B
                                                                                                    ~
                                                                         E−         β(β · E) + c β
                                                                              γ+1
                                                                  +
                                                                    2c

                                                                                                1
where c, e and m are the speed of light, the electric charge and the mass of the
particle, G is the anomalous magnetic moment (≈ 1.8 for protons), β         ~ and γ are the
                                              ~ and E
relativistic velocity and the Lorentz factor, B     ~ are the magnetic and electric field
vectors respectively. η is the EDM coefficient, defined by d ~ p = ηqh̄ ~s.
                                                                   2mc
    Neglecting the magnetic field it becomes
                          "                                                    #
             d~s    e                 1 β  ~ ×E~    η ~        γ ~ ~ ~
                 = ~s × − G +                             E−         β(β · E)
             dt     m               γ+1        c               γ+1
                                                   +                                     (2)
                                                      2c

The storage ring EDM experiment [9] aims to measure the second term of the Equation,
which is proportional to η and the radial electric field Er . That will induce a few nrad/s
of sy for η ≈ 2 × 10−15 , corresponding to EDM of dp ≈ 10−29 e·cm inside a ring with
< Er >≈ 5 MV/m. The other term in the equation is a potential systematic error
source, being enhanced with strong vertical electric field Ey .
    In addition, the spin grows a radial component under certain circumstances. Ne-
glecting magnetic fields and transverse components of the velocity, the dominant term
leading to horizontal spin precession comes from the third term of Equation 1.
                                        "                      #
                            dsr     e             1  βl Er
                                = − sl G +
                            dt      m            γ+1 c
                                                                                        (3)

The angle between the spin and momentum can be calculated by subtracting from this
term the angular momentum vector. In the particle’s rest frame in horizontal plane, its
rate of change approximates to
                                  e         1  βl Er
                           ωa = −     G− 2
                                  m       γ −1 c
                                                                                    (4)

Ideally it is possible to freeze the horizontal spin precession with respect√to the velocity
by injecting the beam with a specific energy, determined by γ0 = 1/G + 1 from
Equation 4. This is the basic idea of “frozen spin method” [1, 4, 5]. For protons, this
condition can be satisfied with p0 ≈ 0.7 GeV/c, coined as “magic momentum”.
    In practice there will be a momentum spread around the magic momentum. Besides,
almost all particles make betatron oscillations due to transverse velocity and their
transverse offset at the time of injection. Nevertheless, their momentum can be made
to oscillate around the magic value by using an RF cavity, so that the average spin
of the beam oscillates horizontally around the momentum vector [6]. There is still a
tiny drift coming from the uncorrected second order term. This tiny drift eventually
makes the radial spin component to grow as much as a radian within the so-called “spin
coherence time”. As a consequence, spin decoherence comes with off-magic momentum.
    The first term of Equation 2 shows that under a vertical electric field, the off-magic
momentum particles will grow a vertical spin component as well. In the storage ring
EDM experiment, this scenario can be encountered as a result of misalignment of
electrostatic elements like deflectors, quadrupoles, etc [8]. Note that, even though the
average < Ey >= 0, the effects are finite.

                                             2
2     Application to the pEDM experiment
A recently published paper [9] describes a storage ring experiment for probing the
EDM of proton at the 10−29 e · cm level. Longitudinally polarized counter-rotating
proton beams will be injected at approximately 0.7 GeV/c and stored for 1000 seconds
inside a 500m long ring. The ring is composed of electrostatic elements with a simple
FODO lattice. The momentum of the beams will be averaged to the magic value by
an RF cavity as explained above. The radial electric field will couple with the EDM
to grow a vertical spin component. The beams will be continuously extracted to the
polarimeter [10] for spin measurement during storage. The total measurement time
will be of order of 107 seconds. The quadrupoles are designed to be 40cm long with
roughly 35 MV/m2 focusing strength.


3     Misplacement of quadrupoles
In case of misplacement of focusing elements, the closed orbit can be shifted both
horizontally and vertically. The vertical offset causes a net vertical electric field locally
in the ring, which causes off-plane spin precession similar to the EDM signal according
to the first term of Equation 2. It also causes an off-magic momentum because of the
vertical motion it introduces. Similarly, the horizontal offset also causes a change in
momentum even if it was magic at the time of injection. Combining those two effects
enhances the false EDM signal.
    We made simulations to study how the spin precession is influenced by the mis-
placement of quadrupoles. The simulations were made with a tracking code based on
fourth order Runge-Kutta integrator to simulate one proton inside alternating gradient
all-electric lattice. The details of the simulation tool are descibed in [6] for a weak
focusing all-electric and [7] for a weak focusing magnetic ring.
    We studied the false EDM signal originating from basically two scenarios: random
and symmetric misplacement of quadrupoles. While η is kept zero in both cases, the
vertical spin component grows much faster than nrad/s rate. On the other hand, this
effect can be suppressed by using several methods, namely RF frequency tuning, using
correction quadrupoles and beam-based quadrupole alignment.

3.1    Random misplacement of quadrupoles
In these simulations we misplaced quadrupoles in horizontal and vertical directions
separately in a random fashion. Keeping the pattern the same, the misplacements
were scaled at various simulations. Figure 1 shows the offset of each quadrupole in
the simulations for the case of < 100µm maximum. While traveling around the ring,
the particle sees a net offset of a few µm on average (Figure 2), as the misaligned
quadrupoles do not necessarily average to zero. This causes a distortion of the closed
orbit.


                                             3
                                                             �����������                              �����������
                                 ����
                                     ���
                                     ���
             �����������������
                                     ���
                                     ���
                                         ��
                                     ���
                                     ���
                                     ���
                                     ���
                                 ����
                                                 �   �       ��       ��
                                                                      �        �   ��       �   �� ��� ���   �   ��   �   ��   �   ��
                                                                                                 ������

Figure 1: The quadrupoles are misaligned between ±100µm both horizontally and
vertically.


                                 ���
                                                                                                      �����������
                                                                                                      �����������
                                 ���

                                  ��
             �����������������




                                 �



                                 ���

                                     ��

                                 ���

                                 ���

                                 ���
                                             ��          ��       ���      ���          �   �� ��� ���       ���      ���      ���
                                                                                             ������

Figure 2: Passing through each quadrupole, the particle sees a net misalignment along
the ring.


                                                                                                4
                          ��



                          ��
              ���������


                          ��



                          ��
                               �   �   ��   �   ��      ��
                                                         �    ���   �   ��   �   ��
                                                     ������

Figure 3: Random quadrupole misalignment makes sy grow quadratically overweight-
ing the EDM signal.

     Figure 3 shows the vertical spin component growing quadratically due to these
misplacements. Note that this effect is enhanced by two factors: The presence of
vertical electric fields and off-magic momentum. The latter also causes accumulation
of radial spin component as shown in Equation 4. One can tune the frequency of the
RF cavity to adjust the momentum of the particle and minimize both sr and sy . Figure
4 shows the effect of RF tuning on sy . There is a specific RF frequency which stops the
drift of the horizontal spin precession and when the spin is aligned with the velocity
direction, sy freezes too. This correction addresses the problem of distortion of the
orbit in general, not limited to the misalignment of the quadrupoles. Therefore, while
not studied in particular, we expect this method to fix the effect of misalignment of all
electrostatic elements like deflectors, sextupoles, etc.
     As an alternative solution, one can use additional weaker quadrupoles for correcting
the false EDM signal. We simulated a particle with longitudinally aligned spin (φ = 0)
and optimized the horizontal position of the correction quadrupole to 2.192 cm which
gives the smallest radial spin precession rate. Then, we set the horizontal position of
the quadrupole, simulated the particle with φ = 450 and minimized the vertical spin
precession rate (false EDM signal) by changing the vertical position of the quadrupole.
Note that the false EDM signal maximizes at φ = 900 and the real EDM signal
minimizes with bigger φ according to Equation 2. For these calculations 450 and 900
do not make a big difference because η is zero.
     Figure 5 shows the vertical spin value changing with φ after the correction of the
quadrupoles. Note the large phase c2 and the strong dependence on the spin angle
(c1 ). This means a small error in the spin polarization causes a large false EDM signal.
     One solution to this issue is to use weaker quadrupoles. The lattice in the refer-


                                                     5
                                                      �   ��
                                                              �

             ���������
                                                          �


                                                      ���
                                                     ����
                                                     ����
                                                                  ��           �   �           ��            �
                                                                                                             �         �
                                                                                                                       �         ���
                                                                                                    ������
                                                     ����
                                                      ���
             ���������




                                                     �


                                                     ����
                                                          �   �
                                                     ����
                                                                  ��           �   �           ��            �
                                                                                                             �         �
                                                                                                                       �         ���
                                                                                                    ������

Figure 4: Precession of sy can be controlled by tuning the RF cavity. Each color
corresponds to a specific RF frequency. There is a specific frequency which minimizes
both sr and sy at the same time.
             vertical spin precession rate (rad/s)




                                                         6x10-7

                                                         5x10-7

                                                         4x10-7

                                                         3x10-7

                                                         2x10-7

                                                         1x10-7
                                                                                                      simulation results
                                                                       0                            c0 + c1 sin(φ0 + c2)

                                                     -1x10-7
                                                                           0       30         60        90       120       150   180
                                                                                       initial spin angle : φ0 (degree)



Figure 5: The dependence of vertical spin component on φ. The coefficients fitting the
data are: c0 = 8.6 × 10−9 , c1 = 5.1 × 10−7 and c2 = −11.5 degrees.



                                                                                                    6
Table 1: Lattice-1 refers to the lattice described in [9]. Lattice-2 has everything the
same except for the quadrupole strength.
                         Quadrupole       Lattice-1 Lattice-2
                                                                k1 (V/m2 )   −3.4 × 107               0
                                                                k2 (V/m )
                                                                        2
                                                                               4.2 × 10   7
                                                                                                     105
                                                                k3 (V/m2 )     3.7 × 107      −2 × 105
                                                                k4 (V/m2 )   −3.2 × 107               0
              vertical spin precession rate (rad/s)




                                                      2x10-9
                                                                                 simulation results
                                                           0                   c0 + c1 sin(φ0 + c2)

                                                      -2x10-9

                                                      -4x10-9

                                                      -6x10-9

                                                      -8x10-9

                                                      -1x10-8
                                                                0     30      60       90      120     150   180
                                                                       initial spin angle : φ0 (degree)



Figure 6: The false EDM signal becomes much smaller with weaker quadrupole
strengths. The fit parameters turn out to be c0 = −5.9 × 10−11 , c1 = −9.1 × 10−9
and c2 = 0.42 degrees.

enced paper [9] (Lattice-1) has four sets of quadrupoles with focusing strengths. We
changed the strength of each quadrupole to achieve a stable storage with a much
smaller value (Lattice-2). These values are chosen arbitrarily to give a stable storage.
The horizontal and vertical emittance are 5 mm·mrad and 0.4 mm·mrad respectively.
Table 1 compares the strength of each quadrupole for the two lattices.
   Figure 6 shows that all the fit parameters decrease considerably with weaker
quadrupoles. For instance, φ0 = 1 mrad ≈ 0.060 leads to about 0.1 nrad/s vertical spin
precession, which is an order of magnitude less than the EDM signal.




                                                                                   7
                                                         k3   k1        k1   k3
                                                    k4                            k4
                                               k3                  +y                  k3
                                          k4                                                k4
                                     k3                                                          k3
                                k4                                                                    k4
                           k3                                                                              k3

                       k4                                                                                   k4

                      k3                                                                                        k3

                      k1                                                                                        k1

                            -x                                                                             +x
                      k1                                                                                        k1

                      k3                                                                                        k3

                       k4                                                                                   k4

                           k3                                                                              k3
                                k4                                                                    k4
                                     k3                                                          k3
                                          k4                                                k4
                                               k3
                                                    k4
                                                         k3
                                                                   -y        k3
                                                                                  k4
                                                                                       k3
                                                              k1        k1




Figure 7: Geometric phase effect arises in some configurations with alternating field.
±x and ±y shows the direction of the misalignment of the quadrupoles. x and y stand
for horizontal and vertical respectively. The average field is zero along the ring, but the
net effect on sy is nonzero due to the order of oscillations in perpendicular directions.
Perpendicular directions do not have to be 900 apart.

3.2   Geometric phase effect
Geometric phase [11], [12] appears in the presence of periodic distortions of field. The
effect originates from the coupling of sy and sr . In such cases, sr is not symmetric while
sy rises and falls. Therefore the amount of rise and fall differ. This can be visualized
imagining Rubick’s cube. One needs to follow the correct order when taking the move
back, otherwise there is a residual effect. The repetition of this at each cycle causes
an accumulation of the vertical spin component [13].
    The spin accumulation in the case of random misplacement of quadrupoles orig-
inates from the geometric phase effect. Each misplaced quadrupole introduces an
additional transverse electric field, changing periodically around the ring, potentially
causing the geometrical phase effect.

3.3   Symmetric distortions along the ring
Figure 7 shows a marginal case of geometric phase effect in a storage ring with four
misplaced quadrupoles at four ends of the ring. The misplacement was introduced
in an alternating fashion to maximize the effect. This way, the particle experiences
spin precession in perpendicular directions one after another in a consecutive fashion.
Eventually, there is a residual amount of sy at each cycle because of the geometrical
phase effect.

                                                                    8
                          �   �
                                  ��������������
                          �   �   ��������������

                          �   �
              ���������   �   �

                          �   �

                          ��

                          ��

                          ��

                          ��
                                    �   �     ���         ���       ���   ��
                                                                          �    ���
                                                          ������

Figure 8: Even though the average quadrupole misalignment is zero along the ring,
the geometric phase effect accumulates a relatively big spin component. In the case
of ±50µm offset, the spin precession rate is ≈ 110 nrad/s and in the case of ±10µm, it
is ≈ 5 nrad/s.

   Figure 8 shows the running average of sy as obtained in simulations with the
configuration shown in Figure 7. The running average after N time steps is defined as
                                                              N−1
                                                          1 X n
                                                   sN
                                                    y =        s
                                                          N n=1 y

with n representing each time step. In one of the cases the offset of each quadrupole
is ±50µm, and in the other case it is ±10µm. Note the quadratic dependence of
the accumulation of sy on quadrupole offset. This comes from the fact that both ra-
dial and vertical oscillations scale locally with the transverse fields, hence transverse
misplacements.
    As seen in the figure, the offset should be aligned to better than 10µm to suppress
false EDM signal due to this effect.
    We repeated the same simulations in the lattice with weaker quadrupoles and an
additional correction quadrupole as explained above. The regular quadrupoles are
given misalignments of 10µm with the configuration shown in Figure 7. Again, the
false EDM signal became negligible compared to the real EDM signal after optimizing
the position of the correction quadrupole (Figure 9).




                                                          9
              vertical spin precession rate (rad/s)
                                                        4x10-8
                                                                               simulation results
                                                             0               c0 + c1 sin(φ0 + c2)

                                                       -4x10-8

                                                       -8x10-8

                                                      -1.2x10-7

                                                      -1.6x10-7

                                                       -2x10-7
                                                                  0   30     60      90    120      150   180
                                                                      initial spin angle : φ0 (degree)



Figure 9: The simulation shows that the geometric phase effect becomes negligible
with weaker quadrupoles. The fit parameters are: c0 = −1.9 × 10−11 , c1 = −1.8 × 10−7 ,
c2 = 0.03 degree.

3.4   Beam-based alignment
It may be possible to align the quadrupoles by separately modulating them at specific
frequencies. We made several simulations with a particle in a lattice with one vertically
misplaced quadrupoles. The misplacement ∆y was set to various values between 5µm
and 30µm. We modulated the strength of the misaligned quadrupole by about 3% at
20 kHz. This modulation was seen in the vertical oscillation of the particle. FFT of the
vertical position y after 4 ms simulation shows peaks at the modulation frequency as
seen in Figure 10. The amplitude of the peak is proportional to the misalignment. The
method requires measuring the vertical position of the beam by a BPM with sub-µm
resolution and then actuating the quadrupole in a way to minimize the peak. This way
a few µm alignment can be achievable.


4     Conclusion
This study investigates the effect of misplacement of quadrupoles on the spin of a
proton inside an all-electric storage ring. Misalignment of quadrupoles is a source
of a systematic error, mimicking the EDM signal. This mainly originates from the
coupling between vertical electric field and the spin.
    We investigated several methods in simulations to solve this issue: namely tuning
the frequency of the RF cavity, and positioning a correction quadrupole in the ring.
While these methods help reducing the effect, lowering the quadrupole strength seems

                                                                              10
                       1.6
                                                  Δy = 5 µm
                       1.4
                                                 Δy = 10 µm
                       1.2                       Δy = 20 µm
                       1.0                       Δy = 30 µm
              y (µm)
                       0.8
                       0.6
                       0.4
                       0.2
                       0.0
                             10   20   30        40   50   60    70   80
                                         Freq (kHz)

Figure 10: Vertical oscillation of the particle has a Fourier component at the modulation
frequency of the quadrupole. The amplitude of the peak becomes smaller with smaller
misalignment.

inevitable for a negligible false EDM signal.
    The quadrupole misplacement should also be kept low for a good control of the
effect. The simulations show that modulation of a misplaced quadrupole modulates the
vertical oscillation of the particle as well. This feature can be exploited to minimize
the misplacement of the quadrupoles. The vertical position should be measured by 0.2
µm to align the relative position of a quadrupole by 5 µm.


References
[1] F.J.M Farley et al, Physical Review Letters 93 (2004) 052001

[2] J.D.Jackson, Classical Electrodynamics, 3rd ed., John Wiley and Sons, NY, USA, 1998,
    p.564

[3] G. W. Bennett et al. (Muon g-2 Collaboration), Physical Review D 73 (2006) 072003

[4] Y.K. Semertzidis et al, arXiv:hep-ph/0012087v1, 2000

[5] Y.K. Semertzidis, arXiv:1110.3378v1 [physics.acc-ph], 2011

[6] S. Haciomeroglu, et al, Nuclear Instruments and Methods in Physics Research A
    743 (2014) 96-102

[7] E.M. Methodiev, et al, Nuclear Instruments and Methods in Physics Research A 797
    (2015) 311,318

                                            11
[8] M. Bai, Y. Dutheil, arXiv:1611:04992v1 [physics.acc-ph], 2016

[9] V. Anastassopoulos et al [EDM Collaboration], Review of Scientific Instruments 87,
    115116 (2016)

[10] Nucl. Instr. Methods Phys. Res. Sect. A 664 (2012), 49

[11] M. Berry, Proc. R. Soc. London, Ser. A 392, 45 (1984)

[12] J.M.Pendelbury, et al, Physical Review A 70 (2004) 032102

[13] Y.F. Orlov, Muon EDM note 26, 2002, available upon request




                                           12
