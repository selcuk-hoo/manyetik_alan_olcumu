# REFERANS: Simultaneous beam-based alignment measurement for multiple magnets (arXiv:2203.14869)

> Kaynak PDF'ten pdftotext ile çıkarıldı (özgünlük karşılaştırması için referans).

---

                                                            Simultaneous beam based alignment measurement for multiple magnets
                                                                                                       Xiaobiao Huang∗
                                                                                 SLAC National Accelerator Laboratory, Menlo Park, CA, 94025
                                                                                                   (Dated: March 29, 2022)
                                                                 We propose a method to simultaneously determine the magnetic centers of multiple quadrupoles
                                                               in a transport line or a storage ring. The method finds the magnet centers by correcting the orbit
                                                               shift due to a change of the quadrupole gradient strengths with orbit correctors. The quadrupoles
                                                               are selected with orbit corrector magnets and beam position monitors in between to ensure that orbit
                                                               correction at the quadrupole locations can be achieved. The correction of the induced orbit shift is
arXiv:2203.14869v1 [physics.acc-ph] 28 Mar 2022




                                                               done by steering the orbit toward the quadrupole centers with correctors, using its response matrix
                                                               with respect to the correctors. The response matrix can be measured or calculated. Simulations
                                                               with a section of the Linac Coherent Light Source (LCLS) II and the SPEAR3 storage ring are
                                                               done to demonstrate the feasibility and performance of the method. It is also experimentally tested
                                                               on SPEAR3. The method can be extended for beam based alignment measurement of nonlinear
                                                               magnets.


                                                                   I.   INTRODUCTION                               step. This could be done manually [5]. A commonly used
                                                                                                                   method is implemented in the Matlab Middle Layer [6],
                                                     Despite the ever improving survey and positioning             for which the quadrupole center offset is found by inter-
                                                  technology, misalignment of magnets in accelerators is           polating the orbit shifts due to the quadrupole gradient
                                                  inevitable. Misalignment causes beam orbit offsets from          variation with respect to the beam orbit to find the zero-
                                                  the centers of the quadrupole and nonlinear (i.e., sex-          crossing [7]. The linear curves of the orbit shift at many
                                                  tupole and octupole) magnets. In storage rings, the              locations vs. the beam orbit at a beam position monitor
                                                  “feed-down” effects of the multipole magnets introduce           (BPM) adjacent to the quadrupole makes a “bow-tie”
                                                  linear optics errors, coupling errors, chromatic errors, and     plot, on which the quadrupole center can be easily recog-
                                                  degradation to nonlinear beam dynamics performances.             nized. The model independent method does not require
                                                  In linacs, the orbit offsets in quadrupoles cause disper-        an accurate lattice model and can find the BPM reading
                                                  sive errors, which leads to emittance dilution. In ad-           corresponding to the quadrupole center on the adjacent
                                                  dition, such orbit offsets complicate the tuning of the          BPM. BPM calibration errors and electrical offsets have
                                                  quadrupoles as any change of gradient will lead to down-         no negative impact on the results.
                                                  stream trajectory shifts. Finding the magnetic centers              A recent progress on the topic is the use of AC excita-
                                                  with beam-based methods and steering the beam through            tion of corrector magnets for beam based alignment [8].
                                                  the magnetic centers of the magnets have many benefits.          The orbit shifts at two selected BPMs are linearly re-
                                                  Beam based alignment (BBA) for quadrupole magnets                lated and the slope of dependence will change when the
                                                  has become a standard practice at modern accelerator             quadrupole strength is varied. The intersection of the two
                                                  facilities.                                                      linear curves, with or without quadrupole strength vari-
                                                     BBA can be done with a model dependent approach               ation, gives the position of the quadrupole center. This
                                                  or a model independent approach. In the model de-                method is fast because beam orbit measurement with AC
                                                  pendent approach, the orbit shift due to a change of             excitation is fast. In addition, horizontal and vertical or-
                                                  the quadrupole gradient is measured and, by the use              bit excitation can be done simultaneously with different
                                                  of a lattice model, the corresponding kick angle at the          driving frequencies. For BBA of quadrupoles in storage
                                                  quadrupole location is calculated, from which the orbit          rings, typically only one magnet is changed at a time.
                                                  offset is obtained [1–3]. The variation of the quadrupole           Reference [9] discusses a few techniques for beam-based
                                                  gradient can be done through a low frequency harmonic            alignment for linacs [10–13]. These methods are similar
                                                  modulation, which leads to an orbit modulation of the            to the methods employed in rings in measuring the tra-
                                                  same frequency [4]. The harmonic modulation reduces              jectory shifts due to a variation of the quadrupole gra-
                                                  noise effects and improves the measurement accuracy.             dients, although in this case, the variation can be intro-
                                                     In the model independent approach, the goal is to find        duced by turning off the selected quadrupoles or measur-
                                                  an orbit through the quadrupole on which a change of             ing the trajectory differences of the electron and positron
                                                  the quadrupole strength does not cause a deflection of           beams (for linear colliders). Most of these methods are
                                                  the beam orbit. This can be achieved by experimentally           model dependent as they solve the quadrupole offsets and
                                                  steering the orbit with a corrector magnet, while observ-        BPM offsets from the measured trajectory shifts with the
                                                  ing the orbit shift by the quadrupole variation at each          use of transfer matrices computed with a model [10–12].
                                                                                                                   However, in one method, the goal is to correct the beam
                                                                                                                   trajectory and simultaneously the trajectory shifts due
                                                                                                                   to the scaling of the strengths of all quadrupoles [13].
                                                  ∗ xiahuang@slac.stanford.edu
                                                                                                                   This method, referred to as dispersion free (DS) correc-
                                                                                                                            2

tion, does not aim at finding the offsets of the individual     ing of new accelerators. It will also enable more frequent
quadrupole magnets, but the minimization of the com-            routine BBA measurements on operating machines.
bined effect of the quadrupole misalignment to beams               The paper is organized as follows: Section II dis-
with energy errors.                                             cusses the method for applications to linacs, including
   Reference [14] proposes a BBA method for quadrupole          detailed descriptions of the theory and simulations for
families on serial power supply. The key idea is to restore     a section of the Linac Coherent Light Source (LCLS)
the orbit after the modulation of quadrupole strengths          II [17]; Section III discusses the method for storage rings
with correctors on or next to the quadrupoles and to            and demonstrates it with the application to the SPEAR3
deduce the initial orbit offsets from the change of correc-     storage ring [18] in both simulation and experiments; Sec-
tor strengths. This method was later tested in experi-          tion IV briefly discusses the special considerations for ap-
ments [15]. A BBA method to address the challenging             plying the method to nonlinear magnets; and Section V
situation in the interaction region of colliders is discussed   gives the conclusions.
in Reference [16].
   In this paper, we propose a beam-based method to
                                                                     II.   P-BBA FOR A TRANSPORT LINE
find the quadrupole magnetic centers for multiple mag-
nets simultaneously. This is achieved by correcting the
orbit shifts due to variations of the quadrupole gradients,                            A.     The method
while the group of quadrupoles are selected to make the
correction possible and easy to do. The method is ap-              In the following we consider BBA for quadrupoles in
plicable to both linacs and storage rings. The proposed         a transport line. Figure 1 is a schematic of the lat-
method is similar to the DS method in correcting the            tice section, which consists of quadrupole magnets, or-
orbit shift induced by quadrupole gradient variations.          bit correctors, and BPMs. The magnetic centers of the
However, in our case, the goal is to determine and regis-       quadrupoles are at ∆i and the beam trajectory passes
ter the quadrupole center offsets with BPMs. Therefore,         through the quadrupoles with position coordinate x̄i , for
the resulting orbit offsets after the correction are not an     i = 1, 2, · · · , N . The quadrupole centers relative to
issue. This is a model independent BBA method, as the           the beam orbit are X̄i = ∆i − x̄i . The quadrupoles
quadrupole offsets found by the method does not require         can be modeled as thin-lens elements. For quadrupole
or depend on a lattice model, even though such a model          i, the nominal integrated gradient is labeled Ki0 , while
could be used to calculate the response matrix (which           the change is labeled ki , and the strengths after changes
could also be measured), and are not affected by BPM            is Ki = Ki0 + ki .
calibration errors or electrical offsets. The pattern of           The beam receives an angular kick by the quadrupole
gradient changes can be properly chosen to facilitate the       when the trajectory is off-centered.           When the
measurements, for example, by alternating the signs of          quadrupole strength is changed, the kick angle will also
gradient variations to keep a stable beam in ring appli-        change. The kick angle change by quadrupole i will be
cations. This method could be extended for nonlinear
magnets in storage rings.                                         ∆φi = ki (∆i − x̄i − ∆x̄i (∆φ1 , ∆φ2 , · · · , ∆φi−1 )), (1)
   This method is also similar to the method discussed
                                                                where ∆x̄i (·) is the trajectory shift at quadrupole i due
in Reference [14] in that both methods use correctors to
                                                                to the kick angle changes by the upstream quadrupoles.
determine the centers of multiple quadrupoles simulta-
                                                                The ∆x̄i (·) term in the kick angle change is nonlinear
neously. However, there are several key differences be-
                                                                with respect to quadrupole strength changes as the ef-
tween the two. First, the proposed method use correc-
                                                                fects of upstream quadrupoles cascade upon each other.
tors to alter the orbit at the quadrupole locations such
                                                                However, we can choose the size of ki to make the nonlin-
that the induced orbit drift is set to zero (or minimized,
                                                                ear terms small. And, as we steer the beam through the
in practice), while in Reference [14] the method aims at
                                                                centers of the quadrupoles, the kick angle changes will
restoring the orbit to before the quadrupole modulation
                                                                diminish and in turn so do these nonlinear terms. In the
is applied. Second, our method registers the quadrupole
                                                                following we drop the ∆x̄i (·) terms, and hence
centers directly with nearby BPMs, while the method in
Reference [14] uses the changes of strengths of the nearby                          ∆φi = ki (∆i − x̄i ).                 (2)
correctors to deduce the orbit offsets at the quadrupole.
Corrector at or near the quadrupoles are required for             The kick angle changes due to gradient variations will
the latter, which cannot always be satisfied, while the         cause the beam trajectory to change. We refer to such
proposed method requires only enough correctors to in-          changes as the induced trajectory (or orbit) shifts. The
dependently change the orbit at the quadrupoles in the          induced trajectory shift at BPM Pi is the position com-
group.                                                          ponent of
   The main benefit of the proposed method is to sub-
                                                                                       Q<P
stantially expedite BBA by parallelizing the process. We                               Xi                       
                                                                                                            0
may refer to the method as parallel BBA (P-BBA). The                         ξ (i) =          M(Pi |Qj )             ,    (3)
                                                                                                           ∆φj
method could have a crucial impact to the commission-                                   j=1
                                                                                                                               3

                                   C1    C2            Q1 C3 P1          Cm QN PM −1           PM



                                   θ1     θ2                θ3 x1        θm         xM −1      xM
                                    FIG. 1. Schematic of an accelerator section for parallel BBA.


where M(Pi |Qj ) is the transfer matrix from quadrupole              where ξ0 = Ak(∆ − x̄0 ) is the induced trajectory shift
Qj to BPM Pi and the condition Q < Pi indicates                      when θ = 0 and
quadrupoles upstream of Pi . If we label the (1,2) ele-
                        (ij)                                                                  ∂ξ
ment of M(Pi |Qj ) as A12 , the trajectory shift at BPMs                                 R≡      = −AkC                     (10)
                                                                                              ∂θ
can be written as
                       Q<P                                           is the response matrix of the induced trajectory shift with
                        Xi
                ξi =
                              (ij)
                             A12 kj (∆j − x̄j ),              (4)    respect to the corrector kick angles.
                       j=1
                                                                        The goal of the P-BBA method is to find corrector
                                                                     kick angles, θ, to set the induced trajectory shift to zero.
which can be written in a matrix form as                             Knowing the response matrix, R, and the measured in-
                                                                     duced trajectory shift, ξ0 , the changes to the corrector
                       ξ = Ak(∆ − x̄),                        (5)    kick angles required to eliminate the induced trajectory
where k is a diagonal matrix whose (j,j) element is kj ,             shift are given by
A is a matrix of dimension M × N with its (i,j) element
          (ij)                                                                         θ = −(RT R)−1 RT ξ0 .                (11)
being A12 and zero if quadrupole Qj is downstream of
BPM Pi , and ∆ and x̄ are vectors formed with ∆j and                 Because the induced trajectory shift is measured at mul-
x̄j , j = 1, 2, · · · , N , respectively.                            tiple BPMs and all measurements have errors, in reality
    Orbit correctors can change the trajectory at the                the goal will not be achieved exactly. Instead, we aim at
quadrupole locations.            The changes can be calcu-           minimizing the induced trajectory shift through a least-
lated using transfer matrices from the correctors to the             square problem, i.e., we minimize
quadrupoles. At quadrupole Qj , the trajectory will be
the position component of                                                                    χ2 = ξ T ξ.                    (12)
                             C<Qj                    
                    (j)
                             X                       0               This can be achieved iteratively. At each iteration,
           x̄(j) = x̄0 +            M(Qj |Cl )          ,     (6)    Eq. (11) can be used to calculate the required changes
                                                     θl
                             l=1
                                                                     to the kick angles toward the next step.
        (j)
where x̄0 is the coordinates at quadrupole Qj when the                  For the scheme to work, the matrix inversion in
correctors are at the initial values (i.e., θl = 0 for l = 1, 2,     Eq. (11) needs to have a unique solution. In other
· · · , m), M(Qj |Cl ) is the transfer matrix from corrector         words, the quadrupoles, correctors, and BPMs should
Cl to quadrupole Qj and the condition C < Qj represents              be chosen to avoid degeneracy in matrices A and C
correctors before the quadrupole. The trajectory at all              (the diagonal matrix km will be non-degenerate as the
quadrupoles can be written in the matrix form as                     quadrupole gradients are changed). The A matrix will
                                                                     be non-degenerate if no two kick patterns by the se-
                       x̄(θ) = x̄0 + Cθ,                      (7)    lected quadrupoles cause the same trajectory shift on
                                                                     the BPMs. This requires at least two BPMs down-
where x̄ is a N -dimensional vector with its component               stream of the last quadrupole and, for any two con-
being the position coordinates at the quadrupoles, x̄0 =             secutive quadrupoles, there is either at least one BPM
x̄(0), C a N × m matrix whose (j,l) element is the (1,               in between, or two BPMs in the space before the next
2) element of M(Qj |Cl ) if Cl is upstream of Qj or zero             quadrupole. The C matrix will be non-degenerate if the
otherwise, and θ is an m-dimensional vector with all the             correctors can steer the beam to the desired trajectory
corrector kick angles as its elements.                               at all quadrupoles. This requires at least one corrector
   Combining Eqs. (5) and (7), we obtain a relationship              upstream of the first quadrupole and, for any two con-
between the induced trajectory shift by the quadrupole               secutive quadrupoles, there are a pair of correctors in the
gradient changes and the kick angles of the correctors,              space upstream or at least one corrector in between.
                                                                        It is preferable to use all correctors and BPMs avail-
                  ξ = Ak(∆ − x̄0 − Cθ),                       (8)
                                                                     able as it helps increase the level of correction preci-
                    = ξ0 + Rθ,                                (9)    sion. Therefore, we only need to select the group of
                                                                                                                         4

quadrupoles for simultaneous BBA measurements. Usu-                                C.   Simulation
ally we can divide all quadrupoles in a beamline into
several groups, each group consisting of quadrupoles             Simulation has been done to test the proposed P-
with a large distance in between, possibly with some          BBA method. The accelerator modeling code Acceler-
quadrupoles skipped. For example, the first, fourth, sev-     ator Toolbox [19] is used for the simulation. The soft
enth, · · · , quadrupoles can be put in one group; the        X-ray linac-to-undulator (LTU) section of the LCLS-II
second, fifth, eighth, · · · in another group, etc. For a     copper linac [17] is used in the study. The number of
long beamline with many quadrupoles, it may be neces-         relevant elements in the line section are listed in Table I,
sary to first divide it into several sections and group the   including two correctors upstream of the section for each
quadrupoles in each section as described in the above.        plane and 5 BPMs downstream of the section.
This is because of the cascading effects of the induced
trajectory shift due to upstream quadrupoles at down-
stream quadrupoles (see the ∆x̄ term in Eq. (1)). We          TABLE I. Elements of the LCLS-II copper soft X-ray LTU
would like the higher order effects to be much smaller        section used in simulation. Two correctors in each plane up-
than the direct effect.                                       stream of the section and 5 BPMs downstream of the section
   The pattern of gradient changes, k, can be a simple        are added to the system.
scaling change to the initial values, if no quadrupole in-                          Parameter        Value
volved is particularly weak. For example, all quadrupole                            Length (m)       372.5
power supplies can be reduced by 5%. A pattern with                           number of quadrupoles 33
                                                                              number of H correctors 16+2
equal changes of integrated gradients but with alternat-
                                                                              number of V correctors 17+2
ing signs can also be used. It is worth noting that if                           number of BPMs      41+5
multiple quadrupoles are on a serial power supply, their
magnetic centers can still be resolved with the proposed
method, as long as there are correctors and BPMs be-
                                                                 Random misalignment errors are first added to the
tween these quadrupoles to detect and correct their indi-
                                                              quadrupoles in the section, with rms offsets of 100 µm
vidual contributions to the induced trajectory shift.
                                                              for both transverse planes. The quadrupole misalign-
                                                              ment causes a distorted beam trajectory (w/ zero ini-
                  B.   Error estimate                         tial launching angle and position). Corrector magnets
                                                              are used to restore the trajectory toward the target
                                                              (x = y = 0 at all BPMs). The rms trajectory errors
   Because we can select the target quadrupoles accord-       are corrected to below 20 µm. The maximum kick angle
ing to the available corrector magnets and BPMs, we           by the correctors for the two planes is 33 µrad (H) and
can correct the trajectory at the quadrupole locations to     61 µrad (V), respectively.
the accuracy of measurements for the induced trajectory
                                                                 We divide the 33 quadrupoles in the LTU section into
shifts by the BPMs. The BPM measurement errors and
                                                              three groups according to their locations. Group 1 con-
the quadrupole center errors are related through Eq. (8).
                                                              sists of quadrupoles 1, 4, 7, · · · , 31; group 2 consists of
If we define
                                                              quadrupoles, 2, 5, 8, · · · , 32; and group 3 consists of
                             ∂ξ                               quadrupoles, 3, 6, 9, · · · , 33. Figure 2 shows the induced
                      RQ ≡       = Ak                (13)
                             ∂∆                               trajectory shift when the strengths of all quadrupoles in
as the response matrix of the induced trajectory shift        group 1 are scaled up by 5%. The trajectory shift is up
with respect to the quadrupole center offsets, the covari-    to 80 µm. Also shown in the figure are the induced tra-
ance matrix of the errors in the measured quadrupole          jectory shift by the linear model (obtained by scaling up
offsets, Σ∆∆ , is related to the BPM measurement errors       the induced trajectory shift of a tiny gradient change).
through                                                       It can be seen that the higher order terms only cause a
                                                              small deviation from the linear model at the downstream
      Σxx ≡ h(ξ − ξ̄)(ξ − ξ̄)T i = RQ Σ∆∆ RTQ ,       (14)    BPMs.
where h·i represents ensemble average over many mea-             The response matrix of the induced trajectory shift
surements and Σxx =diag(σ12 , σ22 , · · · , σM
                                             2
                                               ), with σi ,   with respect to the correctors is calculated with the lat-
i = 1, 2, · · · , M being the error sigma of the BPMs.        tice model. Figure 3 shows the singular values of the
Therefore,                                                    response matrices of the induced trajectory shifts with
                                                              respect to the correctors for both transverse planes for
     Σ∆∆ = (RTQ RQ )−1 RTQ Σxx RQ (RTQ RQ )−1 .       (15)
                                                              the group 1 quadrupoles. While the dimensions of the
If all BPMs have the same measurement error sigma,            response matrices are 46 × 18 and 46 × 19, respectively,
σBPM , we have                                                for the horizontal and vertical planes, there are only 11
                       2                                      modes with substantial singular values. This is because
                Σ∆∆ = σBPM (RTQ RQ )−1 .              (16)
                                                              there are only 11 quadrupoles. The other SV modes
The diagonal elements in the Σ∆∆ matrix give the vari-        would be exactly zero if not for the higher order effects.
ance of the quadrupole center offset measurements.               The correction of the induced trajectory shift is done
                                                                                                                                                                                                              5

                        50                                                                                                0.6

                                                                                                                                                      X          quads 1, 4, 7,...
                                                                                                                          0.4                         Y

                         0
    ( m)




                                                                                                             X, Y (mm)
                                                                                                                          0.2
                                  before BBA
            y




                                  quads 1, 4, 7, ...
    ,




                                                                                                                                  0
            x




                       -50                X, +5%     K/K
                                          Y, +5%     K/K
                                                                                                                          -0.2
                                          X linear
                                          Y linear
                      -100                                                                                                -0.4
                              0      50       100      150   200     250   300    350   400                                           0          50        100     150       200      250   300   350   400
                                                             s (m)                                                                                                            s (m)
                                                                                                                              15
FIG. 2. The induced trajectory shift by a 5% increase of                                                                                              x
the strengths of group 1 quadrupoles. Solid lines show actual                                                                 10                      y
values obtained by tracking. The dashed lines (‘linear’) show
the values scaled up from a 0.1% gradient change by a factor




                                                                                                                 ( rad)
                                                                                                                                  5
of 50.
                                                                                                                                  0
                       105
                                                                                        X
                                                                                                                              -5
                                                                                        Y

                       100
    Singular Values




                                                                                                                          -10
                                                                                                                                      0          50        100     150       200      250   300   350   400
                                                                                                                                                                              s (m)

                      10-5
                                     quads 1, 4, 7,...                                             FIG. 4.   Top: the beam trajectory after correction of the
                                                                                                   induced trajectory shift. Quadrupole locations in the group
                                                                                                   are marked with vertical lines. Bottom: the required kick
                      10-10
                                                                                                   angle changes on the correctors for the correction.
                              0                 5             10             15               20
                                                             Index

                                                                                                   ure 5 shows the comparison of the measured quadrupole
FIG. 3.    Singular values of the horizontal and vertical re-                                      center offsets to the target values. The mean error sig-
sponse matrices of the induced trajectory shift by a 5% in-                                        mas for quadrupole offsets are 22 and 27 µm, for the
crease of the strengths of group 1 quadrupoles with respect
                                                                                                   horizontal and vertical places, respectively. The errors
to the corrector magnets.
                                                                                                   can be reduced by averaging in the measurement of in-
                                                                                                   duced trajectory shifts or increasing the gradient changes
                                                                                                   of quadrupoles.
with Eq. (11), using only the 11 leading singular values in
the matrix inversion calculation. With one iteration, the
rms values of the induced trajectory shift on the 46 BPMs                                                          0.1
are reduced to 0.35 µm and 0.02 µm for the horizontal
                                                                                                      x Q (mm)




                                                                                                                          0
and vertical planes, respectively, when no measurement
                                                                                                                                          quads 1, 4, 7,...
errors are included to the BPMs. A second iteration re-                                                          -0.1
                                                                                                                                                  target            by BBA
duce them further to 5 nm and 0.1 nm, respectively. The                                                          -0.2
required kick angle changes for the correction are mostly                                                          0.2
below 10 µrad. Figure 4 shows the beam trajectory af-
                                                                                                      y Q (mm)




ter the correction of the induced trajectory shift and the                                                                0

required changes to the corrector kick angles.
                                                                                                                 -0.2
  In simulation, the quadrupole center offsets can be
found with the corrected lattice by tracking a particle                                                                       0             50        100         150        200      250   300   350   400
with initial coordinates of x = 0 and x′ = 0 to the                                                                                                                          s (m)

quadrupole locations. The differences with the target
values are on the a few nano-meter level. Essentially, the                                         FIG. 5.    The quadrupole offsets found with the P-BBA
quadrupole centers can be exactly found if there is no                                             method, with BPM error sigma at 5 µm.
BPM measurement errors.
  With the BPM error sigmas set to σBPM = 5 µm, the                                                  On the real machine, we cannot determine the
correction method is repeated 10 times, from which the                                             quadrupole centers by tracking particles to the
error bars to the quadrupole offsets can be found. Fig-                                            quadrupole locations. Instead, we will use the readings
                                                                                                                                                6

of the BPMs near the quadrupoles in the group to rep-                                 a storage ring, an angular kick at any location affects the
resent the center positions of these quadrupoles. For the                             closed orbit everywhere, the A and C matrices are now
LTU section, every quadrupole is next to a BPM and                                    full matrices.
hence the quadrupole center offsets can be accurately                                    Simultaneous changes of the gradients of many
recorded. Simulation with quadrupoles in group 2 and                                  quadrupoles can substantially change the linear optics
group 3 yields similar results. Combining the results from                            of the ring, which could cause significant degradation of
all three groups, the center offsets of all quadrupoles in                            beam lifetime or move the beam across resonance con-
the LTU section are found. Figure 6 shows the com-                                    ditions and in turn cause beam losses. Therefore, we
parison of the BPM offset values found by the P-BBA                                   should choose the quadrupoles carefully and apply a gra-
method and the target offset values at the quadrupoles.                               dient change pattern, k, properly, to ensure the beam
                                                                                      will be stable during and after the gradient changes. For
              0.2                                                                     example, the signs of the gradient changes can be alter-
              0.1                                                                     nated in a sequence of quadrupoles to keep the betatron
   x Q (mm)




                0                                                                     tunes nearly fixed. The number of the quadrupoles in a
              -0.1                                                                    group can be limited to allow a relatively large gradient
              -0.2
                             by BBA (BPM)         target (quads)                      change while keeping the beam stable.
              0.2

              0.1
   y Q (mm)




                0                                                                                          B.   Simulation
              -0.1

              -0.2                                                                       Simulations with the SPEAR3 storage ring are done
                    0   50        100       150     200       250   300   350   400   to demonstrate the application of the P-BBA method
                                                     s (m)                            to storage rings. SPEAR3 is a 3-GeV third generation
                                                                                      synchrotron light source with a circumference of 234 me-
FIG. 6. The quadrupole center offsets registered by BPMs                              ters [18]. The lattice consists of 18 double bend achromat
through the P-BBA method are compared to the target values
                                                                                      (DBA) cells in a racetrack configuration, with 14 stan-
at the quadrupole locations, for all quadrupoles in the LTU
                                                                                      dard cells forming two arcs and 4 matching cells that
section.
                                                                                      flank the two long straight sections. There are a total of
                                                                                      97 quadrupole magnets in the lattice. There are 58 hor-
  Performing P-BBA for all 33 quadrupoles in the LTU
                                                                                      izontal correctors and 56 vertical correctors. Currently
section requires only three times of correction of the in-
                                                                                      56 BPMs are used for beam orbit control.
duced trajectory shifts. Each time the quadrupole gradi-
                                                                                         In the simulation we first introduce random misalign-
ents are varied two to three times. The total time would
                                                                                      ment errors to all quadrupoles with an rms offset of
be substantially less than the current method of making
                                                                                      200 µm for both planes. The orbit is then corrected to
the “bow tie” plot for each individual quadrupole.
                                                                                      below 2 µm at the BPMs with the correctors. We select
                                                                                      a group of 14 quadrupoles for simultaneous BBA as an
                                                                                      example. These are the second QF magnet in each of the
               III.     P-BBA FOR A STORAGE RING
                                                                                      14 standard cells. The QF magnets are 0.35 m long and
                                                                                      the nominal gradients are about 1.9 m−2 . We choose
                                        A.        Method                              to alternate the gradient changes with a +4% change
                                                                                      for the odd number quadrupoles and a −4% change for
   The method of performing simultaneous BBA for mul-                                 the other quadrupoles. The betatron tunes become to
tiple quadrupoles by correcting the induced orbit shift                               νx = 14.070 and νy = 6.153, down from the original val-
(IOS) can be applied to storage rings. Similarly, we select                           ues of νx = 14.106 and νy = 6.177.
a group of quadrupoles that are sufficiently separated,                                  The initial IOS by the gradient changes of the 14 QF
with BPMs and correctors in between, and measure the                                  magnets is shown in Figure 7. Also shown in the plots are
IOS by varying the gradients of these quadrupoles. Cor-                               the expected orbit shift for a linear model (with respect to
rector magnets are used to correct the IOS observed by                                the quadrupole gradients), which is obtained by scaling
the BPMs. Essentially, we are correcting the orbit at the                             up the response of a tiny gradient change by the same
locations of the selected quadrupoles toward the mag-                                 pattern. The differences between the actual orbit shift
netic centers.                                                                        and the linear model reflect the changes to the linear
   The description of the method presented in section II A                            optics of the ring when the quadrupoles are changed.
still largely applies, except now the BPMs measure the                                   The response matrices of the IOS with respect to the
closed orbit, instead of the one-pass trajectory. The ele-                            correctors are calculated with the design lattice model.
ments in the A matrix are now the orbit responses at the                              Figure 8 shows the singular values of the horizontal and
BPMs by the kicks at the quadrupole locations, while the                              vertical response matrices. There are only 14 modes with
elements in the C matrix are the orbit responses at the                               significant singular values as there are 14 quadrupoles
quadrupole locations by the corrector magnets. Since in                               that affect the IOS. The calculated response matrices are
                                                                                                                                                                                             7

                       400                                                                                           0.5
                                                                                                                                  SPEAR3, 14 QF quads
                       200
    ( m)




                                                                                                        X, Y (mm)
                             0
                                                                                                                         0
          y
    ,




                       -200
          x




                                                                         X, +/-4%       K/K
                       -400                                              Y, +/-4%       K/K
                                                                         X linear                                                                                                        X
                                                before BBA
                                                                         Y linear                                                                                                        Y
                       -600                                                                                          -0.5
                                 0        50         100           150        200                                             0               50       100               150       200
                                                           s (m)                                                                                             s (m)
                                                                                                                         30
FIG. 7. The IOS in the SPEAR3 ring when the gradients of
the selected 14 QF quadrupoles are varied by +4% or −4%.                                                                 20

The dashed lines are obtained by varying the gradients by only
                                                                                                                         10
+0.04% or −0.04% and scaling up the orbit shift linearly.




                                                                                                            ( rad)
                                                                                                                         0


used for the correction of the IOS with Eq. (11). After                                                              -10

three iterations of correction, the residual IOS after cor-
                                                                                                                     -20
rection are reduced to sub-micron level. The differences                                                                                                             x         y
between the quadrupole offsets found by BBA and the                                                                  -30
                                                                                                                              0               50       100               150       200
target values are also on the sub-micron level, as shown
                                                                                                                                                             s (m)
in Table II. The beam orbit is changed to go through
the centers of the 14 QF quadrupoles that are varied (see
                                                                                              FIG. 9. Top: the beam orbit after correction of the IOS for
Figure 9). The required corrector kick angles are below
                                                                                              the SPEAR3 example. The locations of the quadrupoles in
30 µrad.                                                                                      the BBA group are marked with vertical lines. Bottom: the
                                                                                              corresponding corrector kick angles.
TABLE II. Standard deviations of residual IOS at BPMs
and the differences between the BBA results and the actual
quadrupole centers (BBA error) after each iteration for the
                                                                                                The BBA results are affected by BPM measurement
SPEAR3 example.
                                                                                              errors. We repeated the P-BBA process 10 times, with
  iteration IOS-X IOS-Y BBA error (X) BBA error (Y)
                                                                                              random errors added to the orbit measurements and a
              µm     µm         µm             µm
      1       23.9   16.5       19.0           15.7
                                                                                              BPM error sigma of 1 µm. The quadrupole center offsets
      2        3.2    2.3        3.2           2.0                                            are compared to the target values in Figure 10. The
      3        0.6    0.3        0.4           0.3                                            average error sigmas of the offsets are 3.4 µm (X) and
                                                                                              5.4 µm (Y) for the two transverse planes, respectively.

                                                                                                              0.2

                                                                                                                     0
                                                                                                 x Q (mm)




                                                                                    X
                                                                                                            -0.2
                                                                                    Y
                                                                                                                              SPEAR3, 14 QF quads
                                                                                                            -0.4
                       10 0
     Singular Values




                                                                                                                                     target        by BBA
                                                                                                            -0.6
                                                                                                             0.5
                                                                                                 y Q (mm)




                                                                                                                     0

                            -5
                       10
                                                                                                            -0.5
                                 0   10        20      30          40    50             60                      0                       50            100                150       200
                                                      Index                                                                                                 s (m)


                                                                                              FIG. 10. The quadrupole center offsets found by the P-BBA
FIG. 8. The singular values of the horizontal and vertical
                                                                                              method are compared to the target values for the SPEAR
response matrices of the IOS by the 14 QF magnets (with a
                                                                                              example. Error bars are by standard deviations of 10 random
scale change of +4% or −4% of alternating signs) with respect
                                                                                              seeds, with BPM noise sigma of 1 µm.
to the correctors.
                                                                                                                                                           8

                         C.        Experiments                                   noticeable differences on the vertical plane, which would
                                                                                 decrease if the vertical IOS correction is improved.
   The P-BBA method has been experimentally tested on
the SPEAR3 storage ring. In the experiment, the same                                                      1
                                                                                                                                               QMS
14 QF magnets as used in simulation are targeted. The




                                                                                         x offset (mm)
                                                                                                                                               PBBA
quadrupole gradients are changed by ±2% in an alter-                                                     0.5

nating pattern.
   Figure 11 shows the IOS measured during three iter-                                                    0
ations of correction. The initial IOS are up to 0.1 mm                                                   0.2
and 0.05 mm, respectively, in the horizontal and vertical




                                                                                     y offset (mm)
                                                                                                         0.1
planes. The conditions for ‘after iteration 1’ and ’before
                                                                                                          0
iteration 2’ are the same, so as ’after iteration 2’ and
                                                                                                     -0.1
’before iteration 3’. The measured IOS for these condi-
                                                                                                     -0.2
tions overlap, which indicate the orbit shifts are repro-                                                0     2   4   6        8    10   12          14
ducible. The rms IOS is reduced from 65.0 µm to 0.6 µm                                                                 BPM for QF2

for the horizontal plane, and from 27.2 µm to 3.0 µm
in the vertical plane. It is noted that in each iteration                        FIG. 12. The quadrupole center offsets from the initial beam
it is an under-correction in the horizontal plane and an                         orbit as measured by P-BBA in experiment are compared to
over-correction in the vertical plane, which could come                          the QMS offsets.
from errors in the corrector strength calibrations. The
over-correction on the vertical plane is 36%. The conver-                          One iteration of IOS correction with two quadrupole
gence would be much faster if we adjust the current to                           gradient modulation (one before, one after) took 32 sec-
kick angle conversion coefficients for the correctors.                           onds. A full correction with 3 iterations should take
                                                                                 about 70 seconds if the two extra intermediate IOS mea-
                                                                                 surements are skipped (14 second each). This is for the
                   Initial             before iter 2        before iter 3
                   after iter 1        after iter 2         after iter 3
                                                                                 group of 14 quadrupoles. It would take less than 5 minute
                                                                                 to find the offsets for all 56 BPMs for SPEAR3. In com-
           0.1
                                                                                 parison, it takes the current method (QMS) 2.5 hrs to
          0.05
                                                                                 complete BBA for the same quadrupoles.
   (mm)




             0
     x




          -0.05
                                                                                  IV.                      P-BBA FOR NONLINEAR MAGNETS IN
           -0.1
                                                                                                                  STORAGE RINGS
          0.05


                                                                                    The BBA approach by correcting the IOS can be ap-
   (mm)




             0                                                                   plied to sextupole and other nonlinear magnets in storage
     y




                                                                                 rings. The centers of several magnets can be found si-
                                               14 QF2 scaled up by   2%          multaneously, provided that varying their strengths does
          -0.05
               0   10             20      30           40       50          60   not cause beam loss and there are enough correctors and
                                       BPM index                                 BPMs to correct the IOS.
                                                                                    Large relative changes of strengths to the nonlinear
FIG. 11. The measured IOS by 14 QF magnets in a SPEAR3                           magnets may be needed to induce large orbit shifts (in
experiment during 3 iterations of correction. The quadrupole                     comparison to BPM errors). The number of nonlinear
gradient changes are ±2% alternately.                                            magnets that can be changed on such a scale while still
                                                                                 keeping a stable beam may be limited. Groups of nonlin-
   After the IOS correction, the quadrupole centers are                          ear magnets and special patterns of changes for them that
marked by the BPMs next to the quadrupoles. The                                  are applicable for P-BBA could be found experimentally.
measurements are repeated 4 times with the same ini-                                The dependence of IOS from variations of nonlinear
tial orbit, from which the error bars can be estimated.                          magnets on the corrector magnets is not linear. Since
Figure 12 shows a comparison of the quadrupole center                            the actual orbit offsets in the nonlinear magnets are not
offsets from the initial orbit measured by P-BBA and                             known, we cannot calculate the response matrix of the
the conventional ’bowtie’ method (quadrupole modula-                             IOS with respect to the correctors with the lattice model.
tion system, or QMS) [7]. The initial orbit is different                         However, the response matrix can be measured on the
from the QMS offset orbit as steering is needed for injec-                       machine for each iteration of the correction. To reduce
tion or user beamlines. For example, the large horizontal                        the measurement time, it may be necessary to reduce the
offset at BPM 2 in the figure is to create a closed orbit                        number of correctors used for the correction of induced
bump at the injection septum. The P-BBA results are                              orbit. For example, if we are trying to determine the cen-
generally close to the QMS results. There are also some                          ter offsets of 20 sextupoles in a large ring with 300 correc-
                                                                                                                           9

tors, there is no need to use all 300 correctors. Instead,      only to BPM precision limitations. The method is ap-
it would be sufficient to choose 20 to 30 properly chosen       plicable to one-pass systems such as linacs and transport
corrector magnets. It may be possible to form combined          lines, as well as storage rings.
orbit correction knobs with all correctors to target the           Simulations were done for a section of the LCLS-II and
orbits at the selected nonlinear magnets, using singular        the SPEAR3 storage ring to demonstrate the method.
value decomposition on model calculated orbit response          In the LCLS-II example, quadrupole gradients are var-
matrix.                                                         ied by 5%. In the SPEAR3 example, the gradients of
   Beam based optimization methods can also be used to          the selected quadrupoles are varied by +4% or −4% in
find the orbit that minimizes the IOS. The Nelder-Mead          an alternating pattern to keep the betatron tunes nearly
simplex method [20] and the robust conjugate direction          fixed. For both cases, the quadrupoles centers are found
search (RCDS) method [21] would be well suited for this         by the method and the error sigmas for the quadrupole
application. Machine learning based optimization algo-          offsets are about 5 times of the BPM error sigma.
rithms, such as the multi-generation Gaussian process              The method was also experimentally tested on
optimizer [22], can also be used.                               SPEAR3. We successfully demonstrated that the IOS
                                                                are reproducible and can be corrected directly with or-
                                                                bit correctors, using model calculated response matrices.
                  V.   CONCLUSION                               The offsets found by the P-BBA method generally agree
                                                                with the conventional method. It is estimated that the
   We proposed a method, P-BBA, to perform beam-                P-BBA method is 30 times faster than the conventional
based alignment measurements for multiple quadrupoles           method.
simultaneously. In the method, quadrupoles in the lattice          Extension of the method to nonlinear magnets in stor-
are properly selected and grouped according to their lo-        age rings is also discussed.
cations relative to the corrector magnets and BPMs. The
orbit shifts induced by a pattern of strength changes of
the selected quadrupoles are measured with BPMs and                         VI.   ACKNOWLEDGEMENTS
corrected with the corrector magnets using the response
matrix method with the aid of singular value decomposi-           This work was supported by the U.S. Department of
tion. After the correction of the IOS, the beam orbit goes      Energy, Office of Science, Office of Basic Energy Sciences,
through the centers of the selected quadrupoles, subject        under Contract No. DE-AC02-76SF00515




                                                                     Section A:(1988)
 [1] P. Röjsel, Nuclear Instruments and Methods in Physics Research Linac’88          pp. 646–648.
                                                                                 Accelerators,  Spectrometers, Detectors and Associated Eq
 [2] R. Brinkmann and M. Boge, in Proceedings of EPAC’94        [11] C. E. Adolphsen, T. L. Lavine, W. B. Atwood, T. M.
     (1994) pp. 938–940.                                             Himel, M. J. Lee, T. Mattison, R. Pitthak, J. T. Seeman,
 [3] K. Endo, H. Fukuma, and F. Q. Zhang, EPAC’96 , 1657             S. H. Willimas, and G. H. Trilling, in Proceedings of
     (1996).                                                         PAC’89 (1989) pp. 977–979.
 [4] I. Barnett, A. Beuret, B. Defining, P. Galbraith, K. Hen-  [12] P. Emma,          R. Carr,          and H.-D. Nuhn,
     richsen, M. Jonker, G. Morpurgo, M. Placidi, R. Schmidt,        Nuclear Instruments and Methods in Physics Research Section A: Ac
     L. Vos, and J. Wenninger, in Proceedings of the Inter-     [13] T.       Raubenheimer           and       R.        Ruth,
     national Workshop on Accelerator Alignment, Tsukuba,            Nuclear Instruments and Methods in Physics Research Section A: Ac
     Japan (1995) pp. 421–426.                                  [14] R. Talman and N. Malitsky, in Proceedings of PAC03
 [5] D. Rice, G. Aharonian, K. Adams, M. Billing,                    (Portland, Oregon, 2003).
     G. Decker, C. Dunnam, M. Giannella, G. Jack-               [15] I. Pinayev, Nuclear Instruments and Methods in Physics Research Se
     son, R. Littauer, B. McDaniel, D. Morse, S. Peck,          [16] G.      H.       Hoffstaetter     and     F.     Willeke,
     L. Sakazaki, J. Seeman, R. Siemann, and R. Talman,              Phys. Rev. ST Accel. Beams 5, 102801 (2002).
     IEEE Transactions on Nuclear Science 30, 2190 (1983).      [17] LCLS-II Final Design Report, Tech. Rep. (SLAC, 2014).
 [6] G. Portmann, J. Corbett, and A. Terebilo, in Proceedings   [18] R. Hettel, in 9th European Particle Accelerator Conference
     of PAC’05 (2005) pp. 4009–4011.                                 (2004).
 [7] G. Portmann, D. Robin,            and L. Schachinger,      [19] A. Terebilo, in Proceedings of the 2001 Particle Acceler-
     PAC’95 , 2693 (1995).                                           ator Conference, Chicago (2001).
 [8] Z. Martı́, G. Benedetti, U. Iriso, and A. Franchi,         [20] J.       A.        Nelder       and       R.       Mead,
     Phys. Rev. Accel. Beams 23, 012802 (2020).                      The Computer Journal 7, 308 (1965),
 [9] P.     Tenenbaum       and   T.     O.    Raubenheimer,         http://oup.prod.sis.lan/comjnl/article-pdf/7/4/308/1013182/7-4-308.
     Phys. Rev. ST Accel. Beams 3, 052801 (2000).               [21] X. Huang, J. Corbett, J. Safranek,           and J. Wu,
[10] T. L. Lavine, J. T. Seeman, W. B. Atwood, T. M. Himel,          Nucl.Instrum.Meth. A726, 77 (2013).
     A. Petersen, and C. E. Adolphsen, in Proceedings of        [22] X.    Huang,       M.    Song,        and    Z.   Zhang,
                                                                     CoRR abs/1907.00250 (2019), arXiv:1907.00250.
